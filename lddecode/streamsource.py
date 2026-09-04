"""Stream frames straight out of a decode, without materialising a .tbc.

Stacking a side currently decodes each capture to its own ~77 GB .tbc and then
reads all of them back to combine.  For a five-capture side that is about
385 GB written, 385 GB read and 77 GB written again - roughly 850 GB of I/O to
produce 77 GB of output - and the stack cannot start until the last decode has
finished.

lockstep() already merges FrameSource iterators by picture number with a
bounded buffer, so the only missing piece is a source that yields frames as it
decodes.  With one of these per capture a whole disc set decodes in step and
stacks on the fly: one file written, and the stacking overlaps the decoding
instead of following it.

The hook is LDdecode._writeout_data(), the point at which a committed field's
sample data would go to the .tbc.  Hooking there rather than at writeout()
matters: by then the field has been through _output_picture() (the commit-time
chroma DG correction) and _write_field() (EFM demodulation, metadata), so the
picture captured here is byte-for-byte the one the .tbc would have held.  The
decode path itself is untouched, which is what makes this safe to use for the
campaign - that path is what the whole thing is validated against.

One deliberate divergence from the indexed path.  TBCFrameSource keys a
frame by its FIRST occurrence in the capture, which is arbitrary when a
capture opens on a short burst before seeking back: on DS8 CommunityNorth,
file frames 0-4 carry pictures 261-265 while the player is still settling
("unable to find any sync pulses"), the capture then seeks back and plays
255 onward properly from file frame 33, and the .tbc keeps the unsettled
opening frames for those five pictures.  A stream keeps its opening
window's dominant run instead (see dominant_run), so it takes the settled
occurrence.  Measured on a 300-frame window, that is the only difference:
89 of the 94 frames both paths carry are byte-identical, and the five that
differ are exactly these.

Note on why this is safe where the older LDFFrameSource was not: its docstring
warns that driving readfield() *after a programmatic seek* leaves field-sync
state inconsistent.  This seeks at most once, at construction, and then reads
sequentially - exactly what a normal CLI decode does.
"""

import os
import numpy as np

from .stack import (Frame, FrameSource, cav_picture, clv_frame, field_vbi,
                    sieve_keys, _do_pairs, SEQ_TOL)


# Frames buffered before anything is emitted.  A capture that was started
# mid-programme and then seeked back to the disc start sweeps the laser across
# tracks, and the frames it crosses carry real, descending picture numbers
# (see sieve_keys).  An indexed .tbc sees the whole side at once and sorts that
# out; a stream has to hold enough of its opening to do the same, because
# lockstep() needs keys in increasing order and cannot be told to go back.
#
# Sized from the 55 captures decoded so far: after sieving they contain 95
# backward jumps in all, and 89 of those are at frame 200 or earlier (median
# 114, p90 168).  256 frames covers every one of them at about 360 MB per
# capture.  The remaining 6 are mid-disc (frames 9935 to 41892) and cost
# nothing: a mid-disc jump is the player replaying pictures this capture has
# already contributed, and dropping the replay is exactly what the indexed
# path's first-occurrence rule does too.
SETTLE_FRAMES = int(os.environ.get("LDSTACK_SETTLE_FRAMES", "256"))

# Frames sieved per window once settled.
CHUNK_FRAMES = int(os.environ.get("LDSTACK_CHUNK_FRAMES", "64"))

# Entries carried either side of a window: the median test looks two frames
# each way (see sieve_keys), so two is what it takes for a window boundary to
# make no difference to the verdict.
CARRY = 2


def dominant_run(pairs, tol=None):
    """The longest run of keys that advances about one per frame.

    The sieve accepts its very first key unconditionally - there is no
    established progression to judge it against yet - so a capture whose
    opening frames are garbage (no sync, a junk VBI read) seeds the run with a
    junk key and only resyncs onto real playback a few frames later.  An
    indexed .tbc never notices: it sorts the whole side, and one stray key is
    just one stray entry.  A stream cannot, because the guard that keeps keys
    increasing for lockstep() would take that junk key as the high-water mark
    and block every real frame below it for the rest of the disc.

    So the opening window keeps only its dominant progression, which on any
    real capture is the playback the rest of the side continues.
    """
    tol = SEQ_TOL if tol is None else tol
    if not pairs:
        return pairs
    runs, cur = [], [pairs[0]]
    for prev, item in zip(pairs, pairs[1:]):
        if 0 <= item[1] - prev[1] <= tol + 1:
            cur.append(item)
        else:
            runs.append(cur)
            cur = [item]
    runs.append(cur)
    return max(runs, key=len)


class StreamingDecodeSource(FrameSource):
    """Decode an .ldf and yield Frames as they are produced.

    Only frames() is supported - there is no random access, because nothing is
    stored.  That is all lockstep() needs.
    """

    def __init__(self, path, system="PAL", seek=None, length=None, cav=True,
                 inputfreq=None, analog_audio=44100, extra_options=None,
                 scratch_dir=None, name=None, efm=False, threads=4):
        from .decoder import LDdecode
        from .utils import make_loader
        from . import utils_logging as logs
        self.path = path
        self.name = name or os.path.splitext(os.path.basename(path))[0]
        self.cav = cav
        self._length = length
        self._efm = efm
        opts = dict(extra_options or {})
        opts.setdefault("threads", threads)
        if efm:
            opts.setdefault("tbc_efm", True)
            opts.setdefault("preEFM", True)

        # The decoder still wants an output base for its metadata and (when
        # asked) its EFM; point it at scratch.  The .tbc it opens there stays
        # empty - _writeout_data never writes to it.
        import tempfile
        from .stack import resolve_scratch
        self._tmpdir = tempfile.mkdtemp(
            prefix="ldstream_", dir=resolve_scratch(scratch_dir, path))
        self._base = os.path.join(self._tmpdir, "cap")

        if logs.logger is None:
            # The decoder calls logger.status(); a bare logging.Logger has no
            # such method, so use the project's own initialiser.
            logs.logger = logs.init_logging(self._base + ".log")
        self.ldd = LDdecode(
            path, self._base, make_loader(path, inputfreq), logs.logger,
            analog_audio=analog_audio, digital_audio=bool(efm), system=system,
            doDOD=True, inputfreq=inputfreq or 40, extra_options=opts,
        )
        # Geometry is only known once a field has been decoded (it depends on
        # the field itself), and the authoritative copy is the capture row the
        # decoder writes to its own .tbc.db - the very row a .tbc.json is built
        # from.  So read it from there, once there is one.
        self.videoParameters = None
        self._pending = []
        self._install_hook()
        if seek is not None:
            self.ldd.seek(0, seek)

    # ------------------------------------------------------------------ #
    def _install_hook(self):
        """Capture each committed field's finalised picture instead of
        writing it to the .tbc.  Everything before this point (chroma DG,
        EFM, the metadata row) has already run normally."""
        from concurrent.futures import Future
        pending = self._pending

        def _writeout_data(fi, picture, audio, f, efm_out=None):
            if isinstance(picture, Future):     # belt and braces; _write_field
                picture = picture.result()      # normally resolves it already
            pending.append((dict(fi), np.asarray(picture), audio))

        self.ldd._writeout_data = _writeout_data

    def _key(self, fi0, fi1, idx, prev_key):
        vbi = field_vbi(fi0) + field_vbi(fi1)
        if self.cav:
            return cav_picture(vbi)
        if 0x88FFFF in vbi:                 # lead-in must not shadow t=0
            return None
        key = clv_frame(vbi)
        if key is None:                     # match TBCFrameSource: carry on
            key = idx if prev_key is None else prev_key + 1
        return key

    # ------------------------------------------------------------------ #
    def _read_frames(self):
        """Drive the decode, yielding (index, key, Frame) in decode order.

        Fields are paired by position, exactly as the .tbc writer lays them
        down and TBCFrameSource reads them back.
        """
        ldd = self.ldd
        half = None
        idx = 0
        prev_key = None
        eof = False
        while not eof:
            try:
                if ldd.readfield() is None:
                    eof = True
            except Exception:
                eof = True
            if eof:
                self._drain_output_lane()
            while self._pending:
                item = self._pending.pop(0)
                if half is None:
                    half = item
                    continue
                if self.videoParameters is None:
                    self.videoParameters = self._read_vp()
                (m0, p0, a0), (m1, p1, a1) = half, item
                half = None
                key = self._key(m0, m1, idx, prev_key)
                idx += 1
                if key is None:
                    continue
                prev_key = key
                audio = None
                if a0 is not None:
                    audio = np.asarray(a0, dtype="<i2").reshape(-1, 2)
                yield idx - 1, key, Frame(
                    key, self._shape(p0), self._shape(p1), m0, m1,
                    _do_pairs(m0), _do_pairs(m1), audio, None,
                )

    def _read_vp(self):
        """videoParameters from the decoder's own capture row, or None until
        the first field has been committed."""
        from .tbcmeta import CAPTURE_QUERY, video_parameters
        conn = getattr(self.ldd, "dbconn", None)
        if conn is None:
            return None
        try:
            row = conn.execute(CAPTURE_QUERY).fetchone()
        except Exception:
            return None
        return video_parameters(row) if row else None

    def _shape(self, pic):
        vp = self.videoParameters or {}
        fh, fw = vp.get("fieldHeight"), vp.get("fieldWidth")
        a = np.asarray(pic)
        if a.ndim == 1 and fh and fw and a.size == fh * fw:
            a = a.reshape(fh, fw)
        return a

    def _drain_output_lane(self):
        """Let the output stage catch up before we call it EOF.

        With -t N the fields trail the commit loop on OrderedOutputLane, so
        the last few have not reached the hook yet when readfield() runs out.
        _finish_output() is the decoder's own drain: it closes the lane in
        submission order and re-raises anything the lane failed with.
        """
        fin = getattr(self.ldd, "_finish_output", None)
        if callable(fin):
            fin()

    # ------------------------------------------------------------------ #
    def frames(self):
        """Yield Frames in increasing key order, sieved exactly the way an
        indexed .tbc capture is.

        The sieve is fed a window at a time.  Two entries either side of each
        window are carried - the leading pair as neighbour context for the
        median test, the trailing pair held back until the next window can
        give them their own neighbours - and the sequence pass's state is
        threaded through, so a window boundary is invisible to it.
        """
        held = {}                   # idx -> Frame, until the sieve rules on it
        win = []                    # [(idx, key)] in play
        nlo = 0                     # leading entries of win that are context
        state = None
        last = None
        emitted = 0
        settled = False

        def emit(pairs):
            nonlocal last, emitted
            for i, k in pairs:
                fr = held.pop(i, None)
                if fr is None or (last is not None and k <= last):
                    continue
                last = k
                emitted += 1
                yield fr

        def advance():
            """Drop what the sieve can no longer change, keep the carry."""
            nonlocal win, nlo
            win = win[max(0, len(win) - (CARRY + 2)):]
            nlo = min(CARRY, len(win))
            for i in [i for i in held if i < win[0][0]]:
                del held[i]

        for idx, key, fr in self._read_frames():
            held[idx] = fr
            win.append((idx, key))
            if not settled:
                if len(win) < SETTLE_FRAMES:
                    continue
                # The opening window is sieved as a whole and emitted in key
                # order, so a start-of-capture seek sweep (see sieve_keys) is
                # resolved before anything reaches lockstep(), which cannot be
                # told to go back.
                keep, state = sieve_keys(win, nhi=CARRY, return_state=True)
                keep = dominant_run(keep)
                for out in emit(sorted(keep, key=lambda p: p[1])):
                    yield out
                advance()
                settled = True
            elif len(win) - nlo >= CHUNK_FRAMES + CARRY:
                keep, state = sieve_keys(win, state=state, nlo=nlo, nhi=CARRY,
                                         return_state=True)
                for out in emit(keep):
                    yield out
                advance()
            if self._length is not None and emitted >= self._length:
                return
        keep, state = sieve_keys(win, state=state, nlo=nlo, return_state=True)
        for out in emit(keep):
            yield out

    # ------------------------------------------------------------------ #
    def efm_path(self):
        p = self._base + ".efm"
        return p if self._efm and os.path.exists(p) and os.path.getsize(p) else None

    def prefm_path(self):
        p = self._base + ".prefm"
        return p if self._efm and os.path.exists(p) and os.path.getsize(p) else None

    def close(self):
        try:
            self.ldd.close()
        except Exception:
            pass
        if getattr(self, "keep_scratch", False):
            return          # the EFM there is still wanted (see _stream_worker)
        import shutil
        shutil.rmtree(getattr(self, "_tmpdir", ""), ignore_errors=True)


# --------------------------------------------------------------------------- #
#  One decode per process
# --------------------------------------------------------------------------- #
# Driving several StreamingDecodeSource generators from lockstep() runs them
# one at a time: next() on a source's generator *is* that capture's decode
# step, so the captures interleave rather than overlap, and decode is
# GIL-bound besides.  Putting each capture in its own process gets the
# concurrency back, and the bounded queue gives the same backpressure
# lockstep's frame buffer does - a capture that runs ahead blocks until the
# others catch up, so memory stays flat.

def _stream_worker(q, path, kw):
    """Decode one capture, pushing frames to the parent."""
    # A decode is hours of CPU; if the stacker dies without getting to
    # close(), these must not be left running.  PDEATHSIG makes the kernel
    # signal us when the parent goes, whatever killed it.
    try:
        import ctypes, signal
        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)   # PR_SET_PDEATHSIG
    except Exception:
        pass
    src = None
    try:
        src = StreamingDecodeSource(path, **kw)
        sent_vp = False
        for fr in src.frames():
            if not sent_vp and src.videoParameters is not None:
                # only known once a field has been decoded, so it goes ahead of
                # the first frame rather than at construction
                q.put(("vp", src.videoParameters))
                sent_vp = True
            q.put(("f", fr.key, fr.f0, fr.f1, fr.meta0, fr.meta1,
                   fr.do0, fr.do1, fr.audio))
        q.put(("efm", src.efm_path(), src.prefm_path(), src._tmpdir))
    except Exception as e:                      # never wedge the parent
        import traceback
        q.put(("error", "%s: %s" % (type(e).__name__, e), traceback.format_exc()))
    finally:
        if src is not None:
            src.keep_scratch = bool(src.efm_path() or src.prefm_path())
            try:
                src.close()
            except Exception:
                pass
        q.put(None)


class ProcessStreamSource(FrameSource):
    """A StreamingDecodeSource running in its own process."""

    def __init__(self, path, queue_depth=6, **kw):
        import multiprocessing as mp
        self.path = path
        self.name = kw.pop("name", None) or \
            os.path.splitext(os.path.basename(path))[0]
        self._efm = (None, None)
        ctx = mp.get_context("spawn")
        self._q = ctx.Queue(maxsize=queue_depth)
        # NOT daemonic: the decode inside the worker spawns its own field
        # workers, and multiprocessing forbids a daemonic process children.
        # close() terminates it, and stack()'s finally clause calls close().
        self._p = ctx.Process(target=_stream_worker, args=(self._q, path, kw))
        self._p.start()

    def frames(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            tag = item[0]
            if tag == "f":
                _, key, f0, f1, m0, m1, do0, do1, audio = item
                yield Frame(key, f0, f1, m0, m1, do0, do1, audio, None)
            elif tag == "vp":
                self.videoParameters = item[1]
            elif tag == "efm":
                self._efm = (item[1], item[2])
                self._worker_scratch = item[3]
            elif tag == "error":
                raise SystemExit("streaming decode of %s failed: %s\n%s"
                                 % (self.path, item[1], item[2]))

    def efm_path(self):
        return self._efm[0]

    def prefm_path(self):
        return self._efm[1]

    def close(self):
        p = getattr(self, "_p", None)
        if p is not None and p.is_alive():
            p.terminate()
            p.join(timeout=10)
        # A Queue holds POSIX semaphores; a side has one of these per capture
        # and a redo has many sides, so they have to go back explicitly or the
        # run ends in a pile of leaked-semaphore warnings.
        q = getattr(self, "_q", None)
        if q is not None:
            try:
                q.close()
                q.join_thread()
            except Exception:
                pass
        d = getattr(self, "_worker_scratch", None)
        if d:
            import shutil
            shutil.rmtree(d, ignore_errors=True)
