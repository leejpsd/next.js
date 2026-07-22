//! Garbage collection for the persistent backend.
//!
//! GC removes tasks that have become unreachable from the live task graph (their persistent
//! `parent_count` reached 0) from both memory and the on-disk cache. The pass runs under the
//! coordinator's GC phase (see
//! [`SnapshotCoordinator::begin_gc`](crate::backend::snapshot_coordinator)) — which excludes normal
//! operations — and is driven as a fully parallel, unbounded job pool
//! (see [`TurboTasksBackend::gc_collect`]).
//!
//! This module holds the GC-specific logic (the job types, the pool driver, per-job teardown, and
//! the pin/unpin bookkeeping) as an `impl TurboTasksBackend`; it is a child of the `backend` module
//! so it reaches the backend's private state (`storage`, `snapshot_coord`) and the GC-only
//! `execute_context_gc` directly. Callers (`snapshot_and_persist`, `stop`, the background job loop,
//! the `Backend` trait's `pin_task_for_gc`/`unpin_task_for_gc`) live in `mod.rs`.

use std::sync::atomic::{AtomicUsize, Ordering};

use turbo_tasks::{TaskId, TurboTasks, scope::scope_unbounded};

use crate::backend::{
    TurboTasksBackend,
    operation::{
        AggregationUpdateQueue, CleanupOldEdgesOperation, ExecuteContext, ExecuteContextImpl,
        OutdatedEdge, TaskGuard,
    },
    storage::{SpecificTaskDataCategory, TaskDataCategory},
    storage_schema::TaskStorageAccessors,
};

/// One unit of GC work: collect a single task. Parallelism is *across* tasks (the unbounded pool in
/// [`TurboTasksBackend::gc_collect`] runs many jobs on different workers); each task's own teardown
/// is sequential. Collecting a task can discover more jobs — a child the cleanup drives to
/// `parent_count == 0` that is itself collectible — which flow straight back into the pool.
struct GcJob(TaskId);

/// Observability counters for one [`TurboTasksBackend::gc_collect`] pass.
#[derive(Default)]
pub(crate) struct GcStats {
    /// Tasks collected (marked soft-deleted).
    pub collected: usize,
    /// Edges torn down across all collected tasks (children + forward-dependency reverse edges).
    pub edges_deleted: usize,
}

impl TurboTasksBackend {
    /// An execute context for the garbage collector that does not take an operation guard. Only
    /// valid while the caller holds the coordinator's GC phase (which provides exclusion); see
    /// [`ExecuteContextImpl::new_for_gc`].
    fn execute_context_gc<'a>(
        &'a self,
        turbo_tasks: &'a TurboTasks<TurboTasksBackend>,
    ) -> impl ExecuteContext<'a> {
        ExecuteContextImpl::new_for_gc(self, turbo_tasks)
    }

    /// Runs a garbage-collection pass under the coordinator's GC phase. Scans the resident map for
    /// collectible tasks (no persistent parent, quiescent, no aggregation edges), re-validates each
    /// under the exclusion, and tears down the ones that are still collectible: scrubbing their
    /// reverse-dependency edges, decrementing their children's `parent_count` (cascading to any
    /// child that reaches 0), removing them from the in-memory map + task_cache, and buffering an
    /// on-disk tombstone for the next persistence commit.
    ///
    /// The pass is fully parallel and unbounded via [`scope_unbounded`]: work is a pool of
    /// [`GcJob`]s (collect a task) and collecting a task may spawn more (a child driven to
    /// `parent_count == 0` that is itself collectible). There are no synchronization barriers
    /// between "levels" of the cascade — discovered work flows straight back into the pool.
    ///
    /// Why this is safe to run concurrently under the GC phase (which excludes normal operations
    /// but not the GC jobs from each other):
    /// - Each job builds its own [`ExecuteContext`] (`execute_context_gc`); the concurrent-lock
    ///   detector is per-context, so jobs on different threads holding different task guards do not
    ///   false-positive.
    /// - The storage map is a sharded dashmap: different tasks hit different shards; same-task
    ///   access is serialized by the shard lock.
    /// - The cascade decrement (`update_and_get_parent_count(-1)`) is a read-modify-write under the
    ///   child's entry write lock, so if two collected parents decrement the same child
    ///   concurrently, exactly one observes the count hit 0 and spawns its collect — no
    ///   double-collect, no lost decrement.
    /// - A collectible task has `parent_count == 0`, so no *surviving* task lists it as a child: a
    ///   task becomes a collect target only after its last persistent parent was itself collected
    ///   (which removed the edge). So a `Collect` never races a decrement of the same task and
    ///   never `ctx.task`-resurrects a task another job just removed.
    /// - Per-job results merge into shared accumulators guarded by a mutex/atomic, touched only on
    ///   the rare collect/retain outcome (not per decrement or per scrub), so they are not a
    ///   contention hot spot.
    ///
    /// `scope_unbounded` runs jobs on the runtime worker threads plus the calling thread, which
    /// drains the whole (growing) pool itself if no helper is scheduled — so this does not depend
    /// on free worker threads (robust on thread-limited runtimes). GC runs from a synchronous
    /// backend context (like `connect_children`, which also fans out onto the scope machinery).
    ///
    /// Returns [`GcStats`] for the pass. The on-disk tombstones are not produced here — collected
    /// tasks are left resident with their `deleted` flag set, and the next snapshot derives the
    /// tombstones from that flag (see `snapshot_and_persist`).
    pub(crate) fn gc_collect(&self, turbo_tasks: &TurboTasks<TurboTasksBackend>) -> GcStats {
        // Seed the pool by scanning the resident map for tasks that pass the cheap
        // `gc_maybe_collectible` pre-filter (a handful of field reads per task under a shard read
        // lock — the same shape as the eviction scan, which proved this is fast). We scan rather
        // than maintain an incremental candidate set: correctness derives entirely from each task's
        // durable `parent_count`, so there's nothing to persist across sessions and nothing to keep
        // in sync (a scan can't miss a task the way a hand-maintained side-set could). `Collect`
        // re-validates each candidate authoritatively under a guard. The scan only sees resident
        // tasks; disk-only garbage is collected after it is next restored.
        //
        // TODO(perf): recycle the task ids of collected tasks. `persisted_task_id_factory`
        // (`IdFactoryWithReuse`) can hand out freed ids, and the persisted `next_free_task_id`
        // high-water mark only grows today, so the id space grows unboundedly across churn even
        // though the task set stays flat. Reuse must happen only AFTER the `save_snapshot` that
        // tombstoned the id has committed (a crash before commit leaves the task on disk — reusing
        // its id would alias it), and must be guarded against resurrection: between removal and
        // commit a `get_or_create_task` for the same type could re-mint the id, and the id must not
        // be handed out while any live `OperationVc`/`DetachedVc` still references it. Feed the
        // recycled ids into `persisted_task_id_factory` so the high-water mark can stop growing.
        let seeds: Vec<GcJob> = self
            .storage
            .gc_collectible_candidates()
            .into_iter()
            .map(GcJob)
            .collect();
        if seeds.is_empty() {
            return GcStats::default();
        }

        // Written once per collected task (not per child/dep), so the atomics are not a hot path.
        let collected = AtomicUsize::new(0);
        let edges_deleted = AtomicUsize::new(0);

        // Each job builds its own GC `ExecuteContext`; see the doc above for the concurrency
        // argument. A job may spawn follow-up jobs (children driven to `parent_count == 0`) that
        // flow straight back into the same pool.
        scope_unbounded(seeds, |spawner, GcJob(task_id)| {
            let mut ctx = self.execute_context_gc(turbo_tasks);
            // `All` restores Data so the edge capture below can read the Data-category dep sets.
            // The collect target came from the resident-map scan, so it must exist.
            let mut task = ctx.task(task_id, TaskDataCategory::All);
            debug_assert!(
                task.is_gc_collectible(),
                "gc: GcJob({task_id}) for a non-collectible task — the seed scan's Meta-resident \
                 `gc_maybe_collectible` filter and the cascade's collectibility check should \
                 guarantee collectibility under the GC phase"
            );

            // Mark the task soft-deleted on the guard we already hold (order relative to the
            // `CleanupOldEdges` run below doesn't matter — `deleted` only affects
            // snapshot/eviction/collectibility, none of which the cleanup consults for `task_id`
            // itself). Rather than remove the task now (a later `ctx.task` on it would resurrect it
            // from disk as a zombie), keep it resident: a later step hard-deletes it.
            task.set_deleted(true);
            if task.new_task() {
                // Never persisted: there is nothing on disk to tombstone, so drop it out of the
                // next snapshot's scan entirely (clearing its modified bits + shard count).
                // Eviction still removes the resident, soft-`deleted` task.
                task.discard_modifications_for_gc_new_task();
            } else {
                // Persisted: force the task into the next snapshot's scan so `process` tombstones
                // its on-disk copy. `deleted` is transient (never persisted), so setting it tracks
                // nothing — explicitly track a meta modification. This matters even in the
                // collectible states that otherwise leave meta clean (e.g. a pinned parentless task
                // then unpinned, or one restored parentless from a prior session).
                let _ = task.track_modification(SpecificTaskDataCategory::Meta, "gc_deleted");
            }
            collected.fetch_add(1, Ordering::Relaxed);

            // Capture all of this task's edges and hand them to the same `CleanupOldEdges`
            // operation a re-executing task uses. Besides dropping each child's `parent_count` and
            // scrubbing forward-dep reverse edges, this propagates the aggregation rebalance
            // (removing this task from its children's `upper` sets) — without it, collected
            // children would keep a dangling upper edge and never become collectible. The op opens
            // `ctx.task(task_id)`, so it must run while `task_id` is still resident.
            let mut old_edges: Vec<OutdatedEdge> = Vec::new();
            old_edges.extend(task.iter_children().map(OutdatedEdge::Child));
            old_edges.extend(
                task.iter_output_dependencies()
                    .map(OutdatedEdge::OutputDependency),
            );
            old_edges.extend(
                task.iter_cell_dependencies()
                    .map(OutdatedEdge::CellDependency),
            );
            old_edges.extend(
                task.iter_cell_dependencies_hashed()
                    .map(|(r, k)| OutdatedEdge::HashedCellDependency(r, k)),
            );
            old_edges.extend(
                task.iter_collectibles_dependencies()
                    .map(OutdatedEdge::CollectiblesDependency),
            );
            drop(task);

            edges_deleted.fetch_add(old_edges.len(), Ordering::Relaxed);
            CleanupOldEdgesOperation::run(
                task_id,
                old_edges,
                AggregationUpdateQueue::new(),
                &mut ctx,
            );

            // `CleanupOldEdges` recorded every child whose persistent `parent_count` reached 0.
            // Re-check collectibility under each child's guard (count 0 alone isn't enough — it
            // could be pinned, a root, or still hold aggregation edges) and spawn a job for the
            // collectible ones. Each child reaches 0 exactly once, so there is no double-queueing.
            // `Meta` suffices — `is_gc_collectible` reads only Meta fields — and a child that turns
            // out collectible re-opens with `All` in its own `Collect` job (restore is cached), so
            // fetching `All` here would only waste a Data restore on the non-collectible children.
            for child in ctx.take_gc_parent_count_zeroed() {
                debug_assert!(
                    !child.is_transient(),
                    "gc: a transient task should never have a persistent parent_count to zero"
                );
                // The child had its `parent_count` decremented during this task's cleanup, so it is
                // resident.
                if ctx.task(child, TaskDataCategory::Meta).is_gc_collectible() {
                    spawner.spawn(GcJob(child));
                }
            }
        });

        GcStats {
            collected: collected.into_inner(),
            edges_deleted: edges_deleted.into_inner(),
        }
    }

    /// Body of [`Backend::pin_task_for_gc`](turbo_tasks::backend::Backend::pin_task_for_gc); the
    /// trait method in `mod.rs` delegates here. See the inline comments for the exclusion and
    /// non-resurrection reasoning.
    pub(super) fn gc_pin(&self, task: TaskId, turbo_tasks: &TurboTasks<TurboTasksBackend>) {
        // Once stopping, GC bookkeeping is irrelevant (the map is torn down in `stop()`), so
        // pin/unpin become no-ops — also safe against handles finalized during shutdown (a
        // `DetachedVc` dropped during Node teardown unpins *after* `stop()` dropped the map).
        if self.stopping.load(Ordering::Acquire) {
            return;
        }
        // An operation-guarded context so pin runs strictly before or after a collection, never
        // concurrently with it. Deadlock-free: no pin caller already holds a guard, and the GC pass
        // never pins.
        let mut ctx = self.execute_context(turbo_tasks);
        // A pin is an in-session reference from outside the tracked graph (`prevent_gc`, or a
        // `DetachedVc` holding the task's `OperationVc` across NAPI), counted like a transient
        // parent's edge: bump `transient_ref_count`, which keeps the task uncollectible and
        // unevictable while > 0. Counting (not a bool) balances each pin against its own unpin.
        //
        // `resident_task` is non-inserting: a pin targets a live reference, so the task must be
        // resident. A missing entry means a pin of an already-collected task (a "zombie
        // `OperationVc`") — surfaced via debug_assert rather than papered over with a blank entry.
        let existed = ctx.resident_task(task);
        debug_assert!(
            existed.is_some(),
            "pin_task_for_gc: task {task} has no resident entry (pinned an already-collected \
             task?)"
        );
        if let Some(mut guard) = existed {
            guard.update_and_get_transient_ref_count(1);
        }
    }

    /// Body of [`Backend::unpin_task_for_gc`](turbo_tasks::backend::Backend::unpin_task_for_gc);
    /// the trait method in `mod.rs` delegates here.
    pub(super) fn gc_unpin(&self, task: TaskId, turbo_tasks: &TurboTasks<TurboTasksBackend>) {
        // See `gc_pin`: no-op once stopping, so handles finalized during shutdown (after the map is
        // dropped) don't underflow the count.
        if self.stopping.load(Ordering::Acquire) {
            return;
        }
        let mut ctx = self.execute_context(turbo_tasks);
        let existed = ctx.resident_task(task);
        debug_assert!(
            existed.is_some(),
            "unpin_task_for_gc: task {task} has no resident entry (unpinned an already-collected \
             task?)"
        );
        if let Some(mut guard) = existed {
            guard.update_and_get_transient_ref_count(-1);
        }
    }

    /// Runs a full GC pass under the GC phase and returns the number of tasks collected (marked
    /// soft-deleted). The tombstones are derived by a subsequent snapshot from the `deleted` flag,
    /// so — unlike before — nothing needs to be threaded to `snapshot_and_evict_for_testing`
    /// (production runs GC inline in `snapshot_and_persist`). Test-only hook; callers must be idle
    /// (no task executing).
    #[doc(hidden)]
    pub fn gc_for_testing(&self, turbo_tasks: &TurboTasks<TurboTasksBackend>) -> usize {
        let _serialize = self.snapshot_in_progress.lock();
        let _gc_phase = self.snapshot_coord.begin_gc();
        self.gc_collect(turbo_tasks).collected
    }
}
