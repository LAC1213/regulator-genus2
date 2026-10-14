"""Optional work budgets and progress records for regulator integration.

Checkpoints run inside continuation and ODE evaluation loops, so a difficult
edge can be interrupted without waiting for the numerical solver to return.
These are cooperative limits; a single external linear algebra call cannot
be interrupted.  Records contain only JSON-serializable values.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from inspect import signature
import math
from time import perf_counter


_active = ContextVar("regulator_integration_monitor", default=None)


def _json(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, complex):
        return [_json(value.real), _json(value.imag)]
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    if hasattr(value, "item"):
        return _json(value.item())
    return str(value)


class IntegrationLimitError(TimeoutError):
    """An integration exceeded its time or work budget.

    This deliberately does not inherit from ``RuntimeError``: numerical
    retry loops must not mistake a budget interruption for a rejected chart.
    ``record`` describes the interrupted edge at the last checkpoint.
    """

    def __init__(self, message, record):
        super().__init__(message)
        self.record = record


class IntegrationMonitor:
    """Collect per-edge diagnostics, with optional cooperative limits.

    Use ``with IntegrationMonitor(...) as monitor:`` around a float64 or
    arbitrary-precision API call.  ``callback(event)`` receives ``start``, ``progress`` and ``end``
    events; progress is throttled to ``progress_interval`` seconds.  A step
    is one continuation attempt or ODE RHS evaluation; auxiliary counters
    (for example Newton iterations) do not count those operations twice.
    Set either limit to ``None`` to disable it.
    """

    def __init__(self, max_edge_seconds=15.0, max_edge_steps=100000,
                 callback=None, progress_interval=1.0, clock=perf_counter):
        if max_edge_seconds is not None and max_edge_seconds <= 0:
            raise ValueError("max_edge_seconds must be positive or None")
        if max_edge_steps is not None and max_edge_steps <= 0:
            raise ValueError("max_edge_steps must be positive or None")
        if progress_interval <= 0:
            raise ValueError("progress_interval must be positive")
        self.max_edge_seconds = max_edge_seconds
        self.max_edge_steps = max_edge_steps
        self.callback = callback
        self.progress_interval = progress_interval
        self.clock = clock
        self.records = []
        self._current = None
        self._phases = []
        self._state = {}
        self._token = None

    def __enter__(self):
        if self._token is not None:
            raise RuntimeError("a monitor cannot be entered twice")
        self._token = _active.set(self)
        return self

    def __exit__(self, *exc):
        _active.reset(self._token)
        self._token = None

    def _snapshot(self, now):
        record = self._current
        return _json({"edge": record["edge"], "degree": record["degree"],
                      "index": record["index"], "state": self._state.copy(),
                      "phase": "/".join(p["name"] for p in self._phases),
                      "seconds": now - record["started"],
                      "steps": record["steps"],
                      "counters": record["counters"].copy()})

    def _emit(self, event, now, **extra):
        if self.callback is not None:
            self.callback(dict(self._snapshot(now), event=event, **extra))

    def checkpoint(self, counter=None, amount=1, **state):
        record = self._current
        if record is None:
            return
        self._state.update(state)
        if counter is not None:
            if counter in ("walk_attempts", "ode_evaluations"):
                record["steps"] += amount
            record["counters"][counter] = record["counters"].get(counter, 0) + amount
            for p in self._phases:
                p["counters"][counter] = p["counters"].get(counter, 0) + amount
        now = self.clock()
        reason = None
        if self.max_edge_steps is not None and record["steps"] > self.max_edge_steps:
            reason = "work budget of %d counted operations" % self.max_edge_steps
        if self.max_edge_seconds is not None and now - record["started"] > self.max_edge_seconds:
            reason = "time budget of %.3g seconds" % self.max_edge_seconds
        if reason is not None:
            snapshot = self._snapshot(now)
            raise IntegrationLimitError(
                "edge %d (degree %s) exceeded its %s in %s; state=%s" %
                (record["index"], record["degree"], reason,
                 snapshot["phase"], snapshot["state"]), snapshot)
        if now - self._last_progress >= self.progress_interval:
            self._last_progress = now
            self._emit("progress", now)

    @contextmanager
    def edge(self, name, degree=None, **state):
        if self._current is not None:
            raise RuntimeError("nested edge diagnostics are not supported")
        now = self.clock()
        record = dict(index=len(self.records), edge=name, degree=_json(degree),
                      started=now, steps=0, counters={}, phases=[], state=_json(state))
        self._current, self._state = record, state.copy()
        self._last_progress = now
        status, error = "ok", None
        try:
            self._emit("start", now)
            yield
            self.checkpoint()
        except BaseException as exc:
            status = "limit" if isinstance(exc, IntegrationLimitError) else "error"
            error = str(exc)
            if isinstance(exc, IntegrationLimitError):
                record["failure"] = exc.record
            raise
        finally:
            now = self.clock()
            record["seconds"] = now - record.pop("started")
            record["status"] = status
            record["last_state"] = _json(self._state)
            if error is not None:
                record["error"] = error
            # _emit needs the running start time until it has built its event.
            record["started"] = now - record["seconds"]
            try:
                self._emit("end", now, status=status, error=error)
            finally:
                record.pop("started")
                self.records.append(record)
                self._current, self._state, self._phases = None, {}, []

    @contextmanager
    def phase(self, name, **state):
        if self._current is None:
            yield
            return
        previous = self._state.copy()
        p = dict(name=name, state=_json(state), counters={}, started=self.clock())
        self._phases.append(p)
        status = "ok"
        try:
            self.checkpoint(**state)
            yield
            self.checkpoint()
        except BaseException:
            status = "error"
            raise
        finally:
            p["seconds"] = self.clock() - p.pop("started")
            p["status"] = status
            p["path"] = "/".join(q["name"] for q in self._phases)
            self._current["phases"].append(p)
            self._phases.pop()
            self._state = previous


def checkpoint(counter=None, amount=1, **state):
    """Count an operation/update state and enforce the active edge budget."""
    monitor = _active.get()
    if monitor is not None:
        monitor.checkpoint(counter, amount, **state)


@contextmanager
def phase(name, **state):
    """Attribute the enclosed work to a named phase of the current edge."""
    monitor = _active.get()
    if monitor is None:
        yield
    else:
        with monitor.phase(name, **state):
            yield


def traced(name, fields=(), is_edge=False):
    """Instrument a function; argument inspection is skipped when disabled."""
    def decorate(function):
        sig = signature(function)

        @wraps(function)
        def wrapped(*args, **kwargs):
            monitor = _active.get()
            if monitor is None:
                return function(*args, **kwargs)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            state = {k: bound.arguments[k] for k in fields}
            if is_edge:
                context = monitor.edge(name, degree=bound.arguments["cov"].N, **state)
            else:
                context = monitor.phase(name, **state)
            with context:
                result = function(*args, **kwargs)
                if (is_edge and isinstance(result, tuple) and len(result) == 2
                        and hasattr(result[1], "dI")):
                    # The trace of a holomorphic differential to P^1 is
                    # zero. Keep this independent per-edge accuracy check
                    # beside the work counters, before global assembly.
                    leg = result[1]
                    monitor._current["trace_periods"] = _json(
                        leg.dI.sum(axis=-1).tolist())
                    monitor._current["trace_log_periods"] = _json(
                        leg.dL.sum(axis=-1).tolist())
                return result
        return wrapped
    return decorate
