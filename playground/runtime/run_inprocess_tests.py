"""Run unit/in-memory ASGI tests without network sockets or a GPU server.

A periodic timer makes asyncio process completed worker callbacks when the
restricted environment cannot deliver its usual socket-based wake-up. This
does not enable network access, bypass permissions, or alter production code.
Only the test client's in-process asyncio loop is changed.
"""
import asyncio
import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

ROOT=Path(__file__).resolve().parent
original_init=TestClient.__init__


def cooperative_loop():
    loop=asyncio.new_event_loop()
    def tick():
        if not loop.is_closed():loop.call_later(.02,tick)
    loop.call_later(.02,tick)
    return loop


def test_client_init(self,*args,**kwargs):
    if kwargs.get('backend','asyncio')=='asyncio':
        options=dict(kwargs.get('backend_options') or {})
        options.setdefault('loop_factory',cooperative_loop)
        kwargs['backend_options']=options
    original_init(self,*args,**kwargs)


class Results:
    def pytest_terminal_summary(self,terminalreporter,exitstatus,config):
        stats=terminalreporter.stats
        result={'verification':'Unit/controller and in-memory ASGI integration; no network/GPU/browser generation',
            'exit_status':int(exitstatus),'passed':len(stats.get('passed',[])),
            'failed':len(stats.get('failed',[])),'errors':len(stats.get('error',[])),
            'skipped':len(stats.get('skipped',[]))}
        (ROOT/'test-results/inprocess-tests.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    TestClient.__init__=test_client_init
    raise SystemExit(pytest.main([str(ROOT/'tests'),'-q'],plugins=[Results()]))
