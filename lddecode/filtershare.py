"""Share RFDecode's static filter coefficients between worker processes.

The demod pool is a spawn pool, so every worker constructs its own RFDecode
and builds its own copy of the filter set.  The coefficients are identical in
every worker - they are a pure function of (constructor options,
DecoderParams) - but each copy occupies its own physical pages, so N workers
put N copies through the shared L3.

Measured on PAL at blocklen 32768: 9.8 MB of filters per instance, of which
8.0 MB never changes after construction.  Only FVideo, Fvideo_eq_auto and
FVideo_rfft are rebuilt at run time (recompute_fvideo / _sync_worker_veq), so
everything else can live in one shared block:

    1 decode at -t 4 (5 processes)    49 MB  ->  8.0 shared + 5 x 1.8 = 17 MB
    3 decodes at -t 4 (15 processes) 147 MB  ->  8.0 shared + 15 x 1.8 = 35 MB

against 36 MiB of L3 on the i9-14900K.

The parent publishes after its own computefilters(), so the block holds the
DecoderParams snapshot the workers are about to adopt; a worker attaches
read-only views over the same physical pages and drops its private copies.
Values are unchanged, so decoded output is bit-identical.
"""

import numpy as np

try:
    from multiprocessing import shared_memory
except ImportError:                                   # pragma: no cover
    shared_memory = None

#: Rebuilt at run time, so never shared: recompute_fvideo() rewrites FVideo and
#: FVideo_rfft on every inverse-MTF adoption, and _sync_worker_veq() rewrites
#: Fvideo_eq_auto when the multiburst EQ servo adopts.
DYNAMIC_FILTERS = frozenset({"FVideo", "Fvideo_eq_auto", "FVideo_rfft"})


def _shareable(filters):
    """(key, array) pairs eligible for the shared block, in a stable order."""
    return sorted(
        (k, v) for k, v in filters.items()
        if isinstance(v, np.ndarray) and k not in DYNAMIC_FILTERS
    )


def publish(filters):
    """Copy the static filters into a SharedMemory block.

    Returns (shm, descriptor).  The caller owns `shm` and must keep a
    reference for as long as any worker may attach, then close()/unlink() it.
    Returns (None, None) when there is nothing to share.
    """
    if shared_memory is None:
        return None, None
    items = _shareable(filters)
    if not items:
        return None, None

    layout, offset = [], 0
    for k, a in items:
        a = np.ascontiguousarray(a)
        layout.append((k, str(a.dtype), a.shape, offset, a.nbytes))
        offset += a.nbytes
    shm = shared_memory.SharedMemory(create=True, size=offset)
    for (k, dt, shape, off, nbytes), (_, a) in zip(layout, items):
        view = np.ndarray(shape, dtype=np.dtype(dt), buffer=shm.buf, offset=off)
        view[...] = np.ascontiguousarray(a)
    return shm, {"name": shm.name, "size": offset, "layout": layout}


def attach(filters, desc):
    """Replace the static entries of `filters` with views on the shared block.

    Returns the attached SharedMemory (keep a reference: the views borrow its
    buffer) or None when nothing was attached.  The private arrays are dropped,
    so the worker's copies become collectable.
    """
    if shared_memory is None or not desc:
        return None
    shm = shared_memory.SharedMemory(name=desc["name"])
    for k, dt, shape, off, _nbytes in desc["layout"]:
        if k not in filters:
            continue
        view = np.ndarray(tuple(shape), dtype=np.dtype(dt),
                          buffer=shm.buf, offset=off)
        view.flags.writeable = False        # nothing may write a shared filter
        filters[k] = view
    return shm


def verify(filters, desc):
    """True when every shared entry matches what the caller holds.

    Used by the unit test and by a paranoid first run; not on the hot path.
    """
    shm = shared_memory.SharedMemory(name=desc["name"])
    try:
        for k, dt, shape, off, _n in desc["layout"]:
            if k not in filters:
                continue
            view = np.ndarray(tuple(shape), dtype=np.dtype(dt),
                              buffer=shm.buf, offset=off)
            if not np.array_equal(view, filters[k]):
                return False
        return True
    finally:
        shm.close()
