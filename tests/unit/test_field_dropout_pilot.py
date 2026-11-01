"""Unit tests for the PAL pilot in Field.dropout_detect_demod.

Inside the sync regions the dropout test relaxes its demod floor to
sync_min (vsync_ire - 60 on PAL, about -103 IRE).  A PAL LaserDisc's
3.75 MHz pilot rides the sync tip in demod, so its troughs reach that floor
and the noise on them was flagged a sample or two at a time on every line.
The sync-region test now reads demod less demod_pilot; the normal-range
test outside the sync regions still reads demod as it is, and NTSC (which
has no pilot channel) is unchanged.

The field here is a stub carrying only what dropout_detect_demod reads:
quiet video at blanking, line starts every LINE samples, an hsync of HSYNC
samples at the start of each line, and no RF or absurd-excursion flags.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from lddecode.field import Field

pytestmark = [pytest.mark.unit, pytest.mark.decode]

LINE = 2560
HSYNC = 188
LINES = 12
IRE0 = 7.1e6
HZ_IRE = 8000.0
VSYNC_IRE = -43.0


def iretohz(ire):
    return IRE0 + HZ_IRE * ire


def stub_field(system="PAL", seed=12345):
    n = LINE * LINES
    names = ["demod", "demod_raw", "demod_05", "demod_burst"]
    if system == "PAL":
        names.append("demod_pilot")
    video = np.recarray(n, dtype=[(name, np.float32) for name in names])
    blank = iretohz(0)
    video["demod"] = blank
    video["demod_raw"] = blank
    video["demod_05"] = blank
    video["demod_burst"] = 0
    if system == "PAL":
        video["demod_pilot"] = 0

    rng = np.random.default_rng(seed)
    tip = iretohz(VSYNC_IRE)
    t = np.arange(HSYNC)
    for line in range(1, LINES - 1):
        s = line * LINE
        video["demod"][s:s + HSYNC] = tip + 1e3 * rng.standard_normal(HSYNC)
        video["demod_05"][s:s + HSYNC] = tip

    rf = SimpleNamespace(
        system=system, delays={"video_rot": 0}, iretohz=iretohz,
        SysParams={"vsync_ire": VSYNC_IRE}, demod_absurd_hz=1.3e7,
    )
    field = SimpleNamespace(
        rf=rf, data={"rfhpf": np.zeros(n, dtype=np.float32), "video": video},
        LT={"hsync": (HSYNC - 20, HSYNC)},
        linelocs=np.arange(LINES + 1, dtype=np.float64) * LINE,
        lineoffset=0, linecount=LINES - 3,
        get_vsync_lines=lambda: [],
    )
    return field, video


def add_pilot(video, peak_ire=65.0):
    """A 3.75 MHz pilot at 40 MHz sampling on every sync tip, in demod and
    in the channel that isolates it; at 65 IRE peak its troughs sit near
    -108 IRE, past sync_min."""
    t = np.arange(HSYNC)
    pilot = peak_ire * HZ_IRE * np.sin(2 * np.pi * 3.75 / 40 * t + 0.3)
    for line in range(1, LINES - 1):
        s = line * LINE
        video["demod"][s:s + HSYNC] += pilot
        video["demod_pilot"][s:s + HSYNC] = pilot


def detect(field):
    return Field.dropout_detect_demod(field)


def test_pilot_troughs_on_the_sync_tips_are_not_dropouts():
    field, video = stub_field()
    add_pilot(video)
    assert np.any(video["demod"] < iretohz(VSYNC_IRE - 60))
    assert not detect(field).any()


def test_pilot_troughs_were_flagged_when_the_pilot_was_left_in():
    # The same tips read without the pilot channel's help (it is zero), as
    # the test used to: the troughs below sync_min are flagged.
    field, video = stub_field()
    add_pilot(video)
    video["demod_pilot"] = 0
    assert detect(field)[LINE * 2:LINE * 2 + HSYNC].any()


def test_a_real_excursion_on_a_sync_tip_is_still_a_dropout():
    field, video = stub_field()
    add_pilot(video)
    s = LINE * 3 + 100
    video["demod"][s:s + 3] = iretohz(-300)
    np.testing.assert_array_equal(np.nonzero(detect(field))[0], np.arange(s, s + 3))


def test_the_pilot_is_not_taken_out_of_the_normal_range_test():
    # Outside the sync regions demod is tested as it is: a pilot-channel
    # reading there must neither clear nor raise a flag.
    field, video = stub_field()
    add_pilot(video)
    low = LINE * 4 + 1000
    video["demod"][low:low + 10] = iretohz(-80)          # below normal_min
    video["demod_pilot"][low:low + 10] = -40 * HZ_IRE    # would lift it into range
    video["demod_pilot"][LINE * 5 + 1000:LINE * 5 + 1010] = 200 * HZ_IRE
    np.testing.assert_array_equal(np.nonzero(detect(field))[0], np.arange(low, low + 10))


def test_ntsc_has_no_pilot_channel_and_is_unchanged():
    field, video = stub_field(system="NTSC")
    s = LINE * 3 + 50
    video["demod"][s:s + 4] = iretohz(-200)
    np.testing.assert_array_equal(np.nonzero(detect(field))[0], np.arange(s, s + 4))
