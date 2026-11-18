# -*- coding: utf-8 -*-
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import animcurve as mod  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _curve():
    d = json.load(open(os.path.join(DATA, "curves.json"), encoding="utf-8"))
    return mod.Curve(d["times"], d["values"], d.get("cyclic", False))


def test_error_within_tolerance():
    c = _curve()
    n = mod.compress(c, 0.05)
    for i in range(200):
        t = c.times[0] + (c.times[-1] - c.times[0]) * i / 199.0
        assert abs(mod.sample(c, t) - mod.sample(n, t)) <= 0.05 + 1e-6


def test_compression_actually_reduces():
    c = _curve()
    n = mod.compress(c, 0.2)
    assert len(n.times) < len(c.times)


def test_cyclic_closes():
    c = _curve()
    n = mod.compress(c, 0.05)
    v0 = mod.sample(n, c.times[0])
    v1 = mod.sample(n, c.times[-1])
    assert abs(v0 - v1) < 0.06


def test_single_keyframe():
    c = mod.Curve([0.0], [0.5])
    n = mod.compress(c, 0.01)
    assert len(n.times) == 1 and abs(mod.sample(n, 3.0) - 0.5) < 1e-9


def test_constant_curve():
    c = mod.Curve([0, 1, 2, 3], [0.7, 0.7, 0.7, 0.7])
    n = mod.compress(c, 0.01)
    assert len(n.times) <= 2


def test_deterministic():
    a = mod.compress(_curve(), 0.05)
    b = mod.compress(_curve(), 0.05)
    assert list(a.times) == list(b.times) and list(a.values) == list(b.values)
