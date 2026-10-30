"""The parallel stack's parent must not wait for ever on a worker that was
killed from outside (OOM killer) and so never reports its blocks."""

import queue
from types import SimpleNamespace

import pytest

from lddecode.stack import _get_result

pytestmark = pytest.mark.unit


class FakeQueue:
    """Hands out the queued items, then reports Empty like a quiet queue."""

    def __init__(self, items):
        self.items = list(items)
        self.timeouts = []

    def get(self, timeout=None):
        self.timeouts.append(timeout)
        if self.items:
            return self.items.pop(0)
        raise queue.Empty


def procs(*codes):
    return [SimpleNamespace(exitcode=c) for c in codes]


def test_result_is_returned_while_workers_run():
    q = FakeQueue([(3, ["r"])])
    assert _get_result(q, procs(None, None), poll=0) == (3, ["r"])


def test_killed_worker_raises_instead_of_hanging():
    q = FakeQueue([])
    with pytest.raises(RuntimeError, match=r"\(1, -9\)"):
        _get_result(q, procs(None, -9, None), poll=0)


def test_quiet_queue_keeps_waiting_while_all_workers_alive_or_finished():
    q = FakeQueue([])
    ps = procs(None, 0)

    def get(timeout=None):
        q.timeouts.append(timeout)
        if len(q.timeouts) < 3:
            raise queue.Empty
        return (0, [])
    q.get = get
    assert _get_result(q, ps, poll=0) == (0, [])
    assert len(q.timeouts) == 3
