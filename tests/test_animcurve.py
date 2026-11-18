# -*- coding: utf-8 -*-
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import animcurve as mod  # noqa: E402
from animcurve.core import max_deviation  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _curve():
    d = json.load(open(os.path.join(DATA, "curves.json"), encoding="utf-8"))
    return mod.Curve(d["times"], d["values"], d.get("cyclic", False))


def _smooth_curve(n=200, cycles=3.0):
    times = [i * 0.05 for i in range(n)]
    values = [math.sin(2 * math.pi * cycles * i / n) + 0.3 * math.sin(6 * math.pi * i / n)
              for i in range(n)]
    return mod.Curve(times, values)


def _mixed_curve(seed, n=500):
    """平坦段与陡变段混排的曲线。"""
    rnd = random.Random(seed)
    base = rnd.uniform(0.0, 10.0)
    times = [i * 0.02 for i in range(n)]
    values = []
    for i in range(n):
        if (i // 50) % 2 == 0:  # 平坦段
            values.append(0.3 + 1e-4 * math.sin(i * 0.7 + base))
        else:  # 陡变段
            values.append(math.sin(i * 0.35 + base) + 0.3 * math.sin(i * 1.7 + base))
    return mod.Curve(times, values)


# ---------------------------------------------------------------- 误差上界

def test_error_within_tolerance():
    c = _curve()
    n = mod.compress(c, 0.05)
    for i in range(200):
        t = c.times[0] + (c.times[-1] - c.times[0]) * i / 199.0
        assert abs(mod.sample(c, t) - mod.sample(n, t)) <= 0.05 + 1e-6


def test_error_bound_exact_on_smooth():
    c = _smooth_curve()
    n = mod.compress(c, 0.02)
    assert max_deviation(c, n) <= 0.02 + 1e-9


def test_compression_actually_reduces():
    # 光滑曲线在 tol 内有真实抽稀空间
    c = _smooth_curve()
    n = mod.compress(c, 0.05)
    assert 2 < len(n.times) < len(c.times)


def test_no_fake_reduction_on_jagged_data():
    # 真实数据逐帧都是极值：tol 不够时不得靠删帧"假装"达标
    c = _curve()
    n = mod.compress(c, 0.2)
    assert max_deviation(c, n) <= 0.2 + 1e-9


def test_not_shrunk_to_endpoints():
    # 非平凡曲线：不得删到只剩首尾来换压缩率
    c = _smooth_curve()
    n = mod.compress(c, 0.01)
    assert len(n.times) > 2
    assert max_deviation(c, n) <= 0.01 + 1e-9


# ---------------------------------------------------------------- 重建 / 折返

def test_no_overshoot_at_reversal():
    c = mod.Curve([0, 1, 2], [0.0, 1.0, 0.0])
    n = mod.compress(c, 0.05)
    for i in range(401):
        t = 2.0 * i / 400
        v = mod.sample(n, t)
        assert -1e-9 <= v <= 1.0 + 1e-9  # 折返点不过冲


def test_violent_reversal():
    # 相邻帧值剧烈折返：有确定输出且误差上界仍成立
    times = [float(i) for i in range(9)]
    values = [0.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 0.0]
    c = mod.Curve(times, values)
    n = mod.compress(c, 0.05)
    assert len(n.times) >= 2
    assert max_deviation(c, n) <= 0.05 + 1e-9
    for i in range(400):
        t = 8.0 * i / 399
        assert abs(mod.sample(n, t)) <= 1.0 + 0.05 + 1e-6


# ---------------------------------------------------------------- 循环

def test_cyclic_closes():
    c = _curve()
    n = mod.compress(c, 0.05)
    v0 = mod.sample(n, c.times[0])
    v1 = mod.sample(n, c.times[-1])
    assert abs(v0 - v1) < 0.06


def test_cyclic_tangents_match():
    # 首尾切线相等：闭合处无速度跳变
    c = _curve()
    n = mod.compress(c, 0.05)
    assert n.tangents[0] == n.tangents[-1]
    # 跨边界采样连续
    t_end = c.times[-1]
    assert abs(mod.sample(n, t_end - 1e-4) - mod.sample(n, t_end + 1e-4)) < 0.05


# ---------------------------------------------------------------- 量化

def test_quantized_output_still_within_tol():
    c = _curve()
    n = mod.compress(c, 0.05)
    # 时间落在量化网格上（端点保持原值），严格递增、无重复
    tq = n.time_step
    assert tq is not None and tq > 0
    for t in n.times[1:-1]:
        assert abs(t / tq - round(t / tq)) < 1e-6
    assert all(n.times[i + 1] > n.times[i] for i in range(len(n.times) - 1))
    assert len(set(n.times)) == len(n.times)
    # 量化后再验误差上界
    assert max_deviation(c, n) <= 0.05 + 1e-9
    for i in range(200):
        t = c.times[0] + (c.times[-1] - c.times[0]) * i / 199.0
        assert abs(mod.sample(c, t) - mod.sample(n, t)) <= 0.05 + 1e-6


# ---------------------------------------------------------------- 边界

def test_single_keyframe():
    c = mod.Curve([0.0], [0.5])
    n = mod.compress(c, 0.01)
    assert len(n.times) == 1 and abs(mod.sample(n, 3.0) - 0.5) < 1e-9


def test_constant_curve():
    c = mod.Curve([0, 1, 2, 3], [0.7, 0.7, 0.7, 0.7])
    n = mod.compress(c, 0.01)
    assert len(n.times) <= 2
    assert abs(mod.sample(n, 1.5) - 0.7) < 0.01 + 1e-9


def test_deterministic():
    a = mod.compress(_curve(), 0.05)
    b = mod.compress(_curve(), 0.05)
    assert list(a.times) == list(b.times) and list(a.values) == list(b.values)


# ---------------------------------------------------------------- 规模

def test_flat_and_steep_both_bounded():
    # 平坦段与陡变段混排：误差上界对两段同样成立
    c = _mixed_curve(0)
    n = mod.compress(c, 0.05)
    assert max_deviation(c, n) <= 0.05 + 1e-9
    assert len(n.times) < len(c.times)  # 平坦段应有明显抽稀


def test_scale_performance():
    # 12 条曲线 × 500 关键帧：压缩 + 重建校验 < 3 秒
    curves = [_mixed_curve(s) for s in range(12)]
    t0 = time.perf_counter()
    total_before = 0
    total_after = 0
    for c in curves:
        n = mod.compress(c, 0.05)
        for i in range(0, len(c.times), 7):  # 重建抽样校验
            t = c.times[i]
            assert abs(mod.sample(c, t) - mod.sample(n, t)) <= 0.05 + 1e-6
        total_before += len(c.times)
        total_after += len(n.times)
    dt = time.perf_counter() - t0
    assert dt < 3.0
    assert total_after < total_before


def test_density_scaling_explainable():
    # 同一信号采样密度翻倍：保留关键帧数基本不变（由信号特征决定），
    # 压缩率近似翻倍 —— 不靠提高误差上界换压缩率
    def mk(n):
        times = [i * 4.0 / (n - 1) for i in range(n)]
        values = [math.sin(2 * math.pi * t) + 0.2 * math.sin(6 * math.pi * t)
                  for t in times]
        return mod.Curve(times, values)

    k500 = len(mod.compress(mk(500), 0.02).times)
    k1000 = len(mod.compress(mk(1000), 0.02).times)
    assert k500 > 2
    assert k1000 <= 2 * k500 + 10
