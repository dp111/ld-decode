"""Stream frames straight out of a decode, without materialising a .tbc.

The stacker currently decodes each capture of a side to its own 77 GB .tbc,
then reads all of them back to combine.  For a five-capture side that is about
385 GB written, 385 GB read and 77 GB written again - roughly 850 GB of I/O to
produce 77 GB of output - and the stack cannot start until the last decode has
finished.

lockstep() already merges FrameSource iterators by CAV picture number with a
bounded buffer, so the only missing piece is a source that yields frames as it
decodes.  With one of these per capture, a whole disc set decodes in step and
stacks on the fly: one file written, and the stacking overlaps the decoding
instead of following it.

The hook is LDdecode.writeout(), which receives (field, fieldinfo, picture,
audio, efm) at commit time - everything a Frame needs - before anything
reaches the disk.  Intercepting there leaves the decode path itself untouched,
which matters because that path is what the whole campaign is validated
against.

Note on why this is safe where the older LDFFrameSource was not: its docstring
warns that driving readfield() *after a programmatic seek* leaves field-sync
state inconsistent.  This seeks at most once, at construction, and then reads
sequentially - which is exactly what the normal CLI decode does.
"""

import os
import numpy as np

from .stack import Frame, FrameSource, cav_picture, clv_frame, field_vbi


class StreamingDecodeSource(FrameSource):
    """Decode an .ldf and yield Frames as they are produced.

    Only frames() is supported - there is no random access, because nothing is
    stored.  That is all lockstep() needs.
    """

    def __init__(self, path, system="PAL", seek=None, cav=True,
                 inputfreq=None, analog_audio=44100, extra_options=None,
                 scratch_dir=None, name=None):
        from .decoder import LDdecode
        self.path = path
        self.name = name or os.path.splitext(os.path.basename(path))[0]
        self.cav = cav
        self._seek = seek
        opts = dict(extra_options or {})

        # The decoder still wants an output base for its metadata bookkeeping;
        # point it at a scratch path and never let it write video (see
        # _install_hook).
        import tempfile
        self._tmpdir = tempfile.mkdtemp(prefix="ldstream_", dir=scratch_dir or "/tmp")
        base = os.path.join(self._tmpdir, "stream")

        from .utils import make_loader
        from . import utils_logging as logs
        if logs.logger is None:
            # The decoder calls logger.status(); a bare logging.Logger has no
            # such method, so use the project's own initialiser.
            logs.logger = logs.init_logging(os.path.join(self._tmpdir, "s.log"))
        opts.setdefault("threads", 4)
        self.ldd = LDdecode(
            path, base, make_loader(path, inputfreq), logs.logger,
            analog_audio=analog_audio, digital_audio=False, system=system,
            doDOD=True, inputfreq=inputfreq or 40, extra_options=opts,
        )
        self.videoParameters = getattr(self.ldd.rf, "videoParameters", None)
        self._pending = []
        self._last_key = None
        self._install_hook()
        if seek is not None:
            self.ldd.seek(0, seek)

    def _install_hook(self):
        """Capture committed fields instead of writing them."""
        ldd = self.ldd
        pending = self._pending

        def writeout(dataset):
            f, fi, picture, audio, efm = dataset
            fi = dict(fi)
            fi["audioSamples"] = 0 if audio is None else int(len(audio) / 2)
            fi["efmTValues"] = 0
            ldd.fieldinfo.append(fi)
            ldd.fields_written += 1
            pending.append((f, fi, picture, audio))

        ldd.writeout = writeout

    def _key(self, fi0, fi1):
        vbi = field_vbi(fi0) + field_vbi(fi1)
        if self.cav:
            return cav_picture(vbi)
        if 0x88FFFF in vbi:            # lead-in must not shadow t=0
            return None
        return clv_frame(vbi)

    def frames(self):
        """Yield Frames in decode order, keyed by picture number."""
        ldd = self.ldd
        prev = None
        while True:
            try:
                if ldd.readfield() is None:
                    break
            except Exception:
                break
            while self._pending:
                f, fi, picture, audio = self._pending.pop(0)
                if prev is None:
                    prev = (f, fi, picture, audio)
                    continue
                (f0, fi0, p0, a0), (f1, fi1, p1, a1) = prev, (f, fi, picture, audio)
                prev = None
                key = self._key(fi0, fi1)
                if key is None:
                    continue
                # lockstep() requires keys in increasing order.  Warm-up and
                # re-decoded fields can arrive out of sequence (measured:
                # 263,264,265 then 228,215,223...), so hold the high-water
                # mark and drop anything that does not advance it.
                if self._last_key is not None and key <= self._last_key:
                    continue
                self._last_key = key
                aud = None
                if a0 is not None:
                    aud = np.asarray(a0, dtype="<i2").reshape(-1, 2)
                yield Frame(
                    key,
                    np.asarray(p0), np.asarray(p1),
                    fi0, fi1,
                    self._dropouts(fi0), self._dropouts(fi1),
                    aud, None,
                )

    @staticmethod
    def _dropouts(fj):
        do = fj.get("dropOuts") or {}
        return list(zip(do.get("fieldLine") or [],
                        do.get("startx") or [], do.get("endx") or []))

    def close(self):
        try:
            self.ldd.close()
        except Exception:
            pass
        import shutil
        shutil.rmtree(getattr(self, "_tmpdir", ""), ignore_errors=True)
