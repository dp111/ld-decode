"""The key sieve both stack input paths share.

An indexed .tbc capture sees a whole side at once; a streaming decode sees a
window at a time.  They must agree on which frames a capture contributes, so
they run the same sieve - these pin the two pathologies it exists for, and
that windowing it changes nothing.
"""

import random

import pytest

from lddecode.stack import sieve_keys

pytestmark = pytest.mark.unit


def sieved(keys, **kw):
    return [k for _, k in sieve_keys(list(enumerate(keys)), **kw)]


def test_ordinary_playback_passes_through():
    assert sieved(list(range(1, 11))) == list(range(1, 11))


def test_isolated_misread_is_rejected():
    """Weak inner-radius RF flipping a picture-number digit: without this the
    bad key shadows the real frame via first-occurrence (EcoDisc S2)."""
    assert sieved([1, 2, 3, 2156, 5, 6, 7]) == [1, 2, 3, 5, 6, 7]


def test_seek_sweep_is_discarded_and_the_resync_adopted():
    """A capture started mid-programme and seeked back sweeps the laser, and
    every track it crosses carries a REAL picture number (CommunitySouth ds1).
    Those keys pass any value test, so only the one-per-frame progression
    separates them from playback."""
    got = sieved([388, 389, 390, 390, 390,
                  340, 290, 240, 190, 90, 30,
                  1, 1, 2, 3, 4, 5, 6])
    assert [k for k in got if k in (340, 290, 240, 190, 90, 30)] == []
    assert got[-6:] == [1, 2, 3, 4, 5, 6]


def test_a_short_run_does_not_confirm_a_resync():
    """SEQ_CONFIRM frames of agreement are required before a new position is
    trusted, or a couple of adjacent misreads would move the sieve."""
    assert 900 not in sieved([1, 2, 3, 900, 901, 5, 6, 7, 8])


# --------------------------------------------------------------------------- #
#  Windowing
# --------------------------------------------------------------------------- #
CARRY, CHUNK = 2, 8


def _windowed(seq):
    """Sieve in windows the way StreamingDecodeSource.frames() does."""
    win, nlo, state, out = [], 0, None, []
    for item in enumerate(seq):
        win.append(item)
        if len(win) - nlo >= CHUNK + CARRY:
            keep, state = sieve_keys(win, state=state, nlo=nlo, nhi=CARRY,
                                     return_state=True)
            out += [k for _, k in keep]
            win = win[max(0, len(win) - (CARRY + 2)):]
            nlo = min(CARRY, len(win))
    keep, _ = sieve_keys(win, state=state, nlo=nlo, return_state=True)
    return out + [k for _, k in keep]


@pytest.mark.parametrize("seed", range(20))
def test_windowing_changes_nothing(seed):
    """A window boundary must be invisible: the carried neighbours feed the
    median test and the threaded state feeds the sequence test."""
    rnd = random.Random(seed)
    seq, k = [], rnd.randint(1, 500)
    for _ in range(rnd.randint(30, 150)):
        if rnd.random() < 0.06:
            seq.append(k + rnd.randint(200, 3000))      # VBI misread
        else:
            k += 1
            seq.append(k)
    assert _windowed(seq) == sieved(seq)


# --------------------------------------------------------------------------- #
#  Opening window
# --------------------------------------------------------------------------- #
def _pairs(keys):
    return list(enumerate(keys))


def test_dominant_run_drops_a_junk_seed_key():
    """The sieve has to accept its first key - there is no progression to
    judge it against yet - so a capture opening on garbage seeds the run with
    a junk key.  For a stream that key would become the monotonic high-water
    mark and block the whole side."""
    from lddecode.streamsource import dominant_run
    got = dominant_run(_pairs([9000, 222, 223, 224, 225, 226, 227]))
    assert [k for _, k in got] == [222, 223, 224, 225, 226, 227]


def test_dominant_run_keeps_ordinary_playback_whole():
    from lddecode.streamsource import dominant_run
    keys = list(range(500, 520))
    assert [k for _, k in dominant_run(_pairs(keys))] == keys


def test_dominant_run_prefers_the_longer_side_of_a_break():
    from lddecode.streamsource import dominant_run
    got = dominant_run(_pairs([10, 11, 12, 700, 701, 702, 703, 704, 705]))
    assert [k for _, k in got] == [700, 701, 702, 703, 704, 705]


# --------------------------------------------------------------------------- #
#  Merging captures
# --------------------------------------------------------------------------- #
class _Keys:
    """A FrameSource that yields nothing but keys."""

    def __init__(self, name, keys):
        self.name = name
        self._keys = keys

    def frames(self):
        for k in self._keys:
            yield type("F", (), {"key": k})()


@pytest.mark.parametrize("a,b", [
    (range(50, 120), range(1, 120)),     # captures starting at different pictures
    (range(1, 60), range(1, 60)),        # identical
    ([1, 2, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7]),   # one capture missing pictures
    (range(1, 40), range(30, 70)),       # barely overlapping
    ([], range(1, 10)),                  # one capture contributes nothing
])
def test_lockstep_emits_the_union(a, b):
    """A picture readable on ANY capture must reach the stacker.

    Merging up to the point where every source has arrived discards whatever
    a capture carried below the latest-starting one's first picture - on
    captures starting at 1 and 50, the first 49 pictures vanished.  Streaming
    sources have no index to recover them from, so the merge itself has to be
    right.
    """
    from lddecode.stack import lockstep
    a, b = list(a), list(b)
    got = [k for k, _ in lockstep([_Keys("A", a), _Keys("B", b)])]
    assert got == sorted(set(a) | set(b))


def test_lockstep_reports_which_captures_carry_each_picture():
    from lddecode.stack import lockstep
    out = dict(lockstep([_Keys("A", [1, 2]), _Keys("B", [2, 3])]))
    assert sorted(out) == [1, 2, 3]
    assert sorted(out[1]) == ["A"] and sorted(out[2]) == ["A", "B"] \
        and sorted(out[3]) == ["B"]
