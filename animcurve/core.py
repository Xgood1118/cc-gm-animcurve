# -*- coding: utf-8 -*-
"""animcurve：误差上界内抽稀、切线重建、量化的动画曲线压缩器。

重建使用单调三次 Hermite（PCHIP 风格切线 + 割线钳制），
折返点切线归零，保证段内单调、不过冲；循环曲线首尾切线强制相等。
"""

from __future__ import annotations

import bisect
import json
import math

__all__ = ["Curve", "compress", "sample", "save"]

_TIME_QUANT = 1e-6   # 时间量化步长上限（实际按 tol 自适应收紧）
_VALUE_QUANT = 1e-5  # 数值量化步长上限


# ---------------------------------------------------------------- 切线

def _pchip_end(h0, h1, d0, d1):
    """非循环端点的单侧切线（PCHIP 规则，保证单调）。"""
    m = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
    if m * d0 <= 0.0:
        return 0.0
    if d0 * d1 < 0.0 and abs(m) > 3.0 * abs(d0):
        return 3.0 * d0
    return m


def _monotone_tangents(times, values, cyclic):
    """PCHIP 单调切线；cyclic 时首尾切线强制相等（闭合处无速度跳变）。"""
    n = len(times)
    if n == 1:
        return [0.0]
    h = [times[i + 1] - times[i] for i in range(n - 1)]
    d = [(values[i + 1] - values[i]) / h[i] for i in range(n - 1)]
    if n == 2:
        return [d[0], d[0]]
    m = [0.0] * n
    for i in range(1, n - 1):
        if d[i - 1] * d[i] <= 0.0:
            m[i] = 0.0  # 局部极值（折返点）：切线归零，不过冲
        else:
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    if cyclic:
        # 闭合：首（=尾）切线由末段与首段割线共同决定
        if d[-1] * d[0] <= 0.0:
            m0 = 0.0
        else:
            w1 = 2.0 * h[0] + h[-1]
            w2 = h[0] + 2.0 * h[-1]
            m0 = (w1 + w2) / (w1 / d[-1] + w2 / d[0])
        m[0] = m[-1] = m0
    else:
        m[0] = _pchip_end(h[0], h[1], d[0], d[1])
        m[-1] = _pchip_end(h[-1], h[-2], d[-1], d[-2])
    return m


def _clamp_tan(m, d):
    """把切线钳制到 [0, 3d]（符号随 d），保证所在段单调。"""
    if m * d <= 0.0:
        return 0.0
    lim = 3.0 * abs(d)
    return max(-lim, min(lim, m))


# ---------------------------------------------------------------- 曲线

class Curve:
    """动画曲线：关键帧 + 单调三次 Hermite 切线。"""

    def __init__(self, times, values, cyclic=False):
        """动画曲线"""
        times = [float(t) for t in times]
        values = [float(v) for v in values]
        if len(times) != len(values):
            raise ValueError("times 与 values 长度不一致")
        if not times:
            raise ValueError("至少需要一个关键帧")
        for i in range(1, len(times)):
            if not times[i] > times[i - 1]:
                raise ValueError("时间戳必须严格递增")
        self.times = times
        self.values = values
        self.cyclic = bool(cyclic)
        self.tangents = _monotone_tangents(times, values, self.cyclic)
        self.time_step = None   # compress 输出的时间量化步长
        self.value_step = None  # compress 输出的数值量化步长


def _hermite(t, t0, v0, m0, t1, v1, m1):
    h = t1 - t0
    s = (t - t0) / h
    s2 = s * s
    s3 = s2 * s
    return ((2.0 * s3 - 3.0 * s2 + 1.0) * v0 + (s3 - 2.0 * s2 + s) * h * m0
            + (-2.0 * s3 + 3.0 * s2) * v1 + (s3 - s2) * h * m1)


def sample(curve, t):
    """按时间采样"""
    times = curve.times
    n = len(times)
    if n == 1:
        return curve.values[0]
    t = float(t)
    if curve.cyclic:
        period = times[-1] - times[0]
        if period > 0.0:
            t = times[0] + (t - times[0]) % period
    if t <= times[0]:
        return curve.values[0]
    if t >= times[-1]:
        return curve.values[-1]
    i = bisect.bisect_right(times, t) - 1
    return _hermite(t, times[i], curve.values[i], curve.tangents[i],
                    times[i + 1], curve.values[i + 1], curve.tangents[i + 1])


# ---------------------------------------------------------------- 误差（精确）

def _seg_coeffs(t0, v0, m0, t1, v1, m1):
    """Hermite 段 → 关于归一化参数 s 的三次系数 (a3, a2, a1, a0)。"""
    h = t1 - t0
    return (2.0 * (v0 - v1) + h * (m0 + m1),
            3.0 * (v1 - v0) - h * (2.0 * m0 + m1),
            h * m0,
            v0)


def _shift_powers(a3, a2, a1, a0, scale, shift):
    """s = scale*u + shift 换元，返回关于 u 的系数 (c3, c2, c1, c0)。"""
    return (a3 * scale ** 3,
            (3.0 * a3 * shift + a2) * scale * scale,
            ((3.0 * a3 * shift + 2.0 * a2) * shift + a1) * scale,
            ((a3 * shift + a2) * shift + a1) * shift + a0)


def _verify(a, b):
    """max |sample(a,t)-sample(b,t)|（t ∈ a 的定义域），精确求三次误差函数临界点。

    返回 (最大偏差, 取到最大偏差的时间)。
    """
    if len(a.times) == 1:
        t = a.times[0]
        return abs(a.values[0] - sample(b, t)), t
    tb = b.times
    nb = len(tb)
    worst = -1.0
    worst_t = a.times[0]
    for i in range(len(a.times) - 1):
        t0, t1 = a.times[i], a.times[i + 1]
        ha = t1 - t0
        ca = _seg_coeffs(t0, a.values[i], a.tangents[i],
                         t1, a.values[i + 1], a.tangents[i + 1])
        # b 在此区间内可能跨段（量化后关键时间轻微移动）→ 按 b 的关键时间再切分
        cuts = [t0]
        cuts.extend(tb[bisect.bisect_right(tb, t0):bisect.bisect_left(tb, t1)])
        cuts.append(t1)
        for k in range(len(cuts) - 1):
            lo, hi = cuts[k], cuts[k + 1]
            if hi <= lo:
                continue
            pa = _shift_powers(ca[0], ca[1], ca[2], ca[3],
                               1.0 / ha, (lo - t0) / ha)
            mid = 0.5 * (lo + hi)
            j = bisect.bisect_right(tb, mid) - 1
            if j < 0:
                pb = (0.0, 0.0, 0.0, b.values[0])
            elif j >= nb - 1:
                pb = (0.0, 0.0, 0.0, b.values[-1])
            else:
                cb = _seg_coeffs(tb[j], b.values[j], b.tangents[j],
                                 tb[j + 1], b.values[j + 1], b.tangents[j + 1])
                pb = _shift_powers(cb[0], cb[1], cb[2], cb[3],
                                   1.0 / (tb[j + 1] - tb[j]),
                                   (lo - tb[j]) / (tb[j + 1] - tb[j]))
            e3 = pa[0] - pb[0]
            e2 = pa[1] - pb[1]
            e1 = pa[2] - pb[2]
            e0 = pa[3] - pb[3]
            L = hi - lo
            cands = [0.0, L]
            # 导数 3*e3*u^2 + 2*e2*u + e1 的实根
            A = 3.0 * e3
            B = 2.0 * e2
            if A != 0.0:
                disc = B * B - 4.0 * A * e1
                if disc > 0.0:
                    r = math.sqrt(disc)
                    cands.append((-B - r) / (2.0 * A))
                    cands.append((-B + r) / (2.0 * A))
            elif B != 0.0:
                cands.append(-e1 / B)
            for u in cands:
                if 0.0 <= u <= L:
                    err = abs(((e3 * u + e2) * u + e1) * u + e0)
                    if err > worst:
                        worst = err
                        worst_t = lo + u
    return worst, worst_t


def max_deviation(a, b):
    """两条曲线在 a 的定义域上的最大逐点偏差（精确）。"""
    return _verify(a, b)[0]


# ---------------------------------------------------------------- 抽稀

def _seg_err(curve, a, e, src_tan):
    """候选段 a→e（Hermite，切线按段割线钳制）对中间原始关键帧的最大误差。"""
    times, values = curve.times, curve.values
    t0, t1 = times[a], times[e]
    h = t1 - t0
    va, ve = values[a], values[e]
    d = (ve - va) / h
    m0 = _clamp_tan(src_tan[a], d) * h
    m1 = _clamp_tan(src_tan[e], d) * h
    worst = 0.0
    for i in range(a + 1, e):
        s = (times[i] - t0) / h
        s2 = s * s
        s3 = s2 * s
        est = ((2.0 * s3 - 3.0 * s2 + 1.0) * va + (s3 - 2.0 * s2 + s) * m0
               + (-2.0 * s3 + 3.0 * s2) * ve + (s3 - s2) * m1)
        err = abs(est - values[i])
        if err > worst:
            worst = err
    return worst


def _thin(curve, tol_dec, src_tan):
    """贪心一趟抽稀：段内误差超界就把前一帧固定为关键帧。O(Σ 段长²)。"""
    n = len(curve.times)
    keep = [0]
    a = 0
    e = 2
    while e < n:
        if _seg_err(curve, a, e, src_tan) <= tol_dec:
            e += 1
        else:
            a = e - 1
            keep.append(a)
            e = a + 2
    keep.append(n - 1)
    return keep


def _clamp2(m, dL, dR):
    """按左右两侧割线钳制切线：异号（折返）归零，同号限幅 3*min。"""
    if dL is None and dR is None:
        return 0.0
    if dL is None:
        return _clamp_tan(m, dR)
    if dR is None:
        return _clamp_tan(m, dL)
    if dL * dR <= 0.0:
        return 0.0
    if m * dL <= 0.0:
        return 0.0
    lim = 3.0 * min(abs(dL), abs(dR))
    return max(-lim, min(lim, m))


def _fit_tangents(curve, keep, src_tan):
    """保留关键帧的切线：继承原曲线切线，再按压缩后的割线钳制保证单调。

    循环曲线首尾切线统一（由首段与末段割线合成），闭合处无速度跳变。
    """
    times, values = curve.times, curve.values
    k = len(keep)
    if k == 1:
        return [0.0]
    sec = [(values[keep[j + 1]] - values[keep[j]])
           / (times[keep[j + 1]] - times[keep[j]]) for j in range(k - 1)]
    out = []
    for j in range(k):
        dL = sec[j - 1] if j > 0 else None
        dR = sec[j] if j < k - 1 else None
        out.append(_clamp2(src_tan[keep[j]], dL, dR))
    if curve.cyclic and k > 2:
        h_first = times[keep[1]] - times[keep[0]]
        h_last = times[keep[-1]] - times[keep[-2]]
        if sec[0] * sec[-1] <= 0.0:
            m0 = 0.0
        else:
            w1 = 2.0 * h_first + h_last
            w2 = h_first + 2.0 * h_last
            m0 = (w1 + w2) / (w1 / sec[-1] + w2 / sec[0])
        out[0] = out[-1] = m0
    return out


def _quantize(curve, tq, vq):
    """时间/数值量化到网格；端点时间保持原值以免定义域漂移。

    量化不改变关键帧顺序、不产生重复时间戳（碰撞时用 nextafter 微顶开）。
    """
    ts = curve.times
    prev = None
    for i in range(len(ts)):
        if 0 < i < len(ts) - 1:
            q = round(ts[i] / tq) * tq
            if q <= prev:
                q = math.nextafter(prev, math.inf)
            ts[i] = q
        prev = ts[i]
    curve.values = [round(v / vq) * vq for v in curve.values]


def _build(curve, keep, src_tan, tq, vq):
    times, values = curve.times, curve.values
    out = Curve([times[i] for i in keep], [values[i] for i in keep], curve.cyclic)
    out.tangents = _fit_tangents(curve, keep, src_tan)
    _quantize(out, tq, vq)
    return out


def _worst_insertable_key(curve, out, keep_set):
    """找可插回的原始关键帧：最差区间端点优先，否则按关键帧处误差最大者。"""
    _, t = _verify(curve, out)
    times = curve.times
    n = len(times)
    i = bisect.bisect_right(times, t) - 1
    i = min(max(i, 0), n - 2)
    for j in (i, i + 1):
        if j not in keep_set:
            return j
    best = -1.0
    best_j = None
    for j in range(n):
        if j in keep_set:
            continue
        e = abs(sample(out, times[j]) - curve.values[j])
        if e > best:
            best = e
            best_j = j
    return best_j


def compress(curve, tol):
    """在误差上界内抽稀关键帧"""
    tol = float(tol)
    if tol < 0.0:
        raise ValueError("tol 必须非负")
    times, values = curve.times, curve.values
    n = len(times)
    src_tan = list(curve.tangents)
    if n <= 2 or tol == 0.0:
        # 单帧/双帧/零容差：原样返回（确定输出）
        out = Curve(list(times), list(values), curve.cyclic)
        out.tangents = src_tan
        return out

    # 自适应量化步长：量化引入的误差预算 ≤ 0.2*tol
    max_m = max(abs(m) for m in src_tan)
    min_h = min(times[i + 1] - times[i] for i in range(n - 1))
    vq = min(_VALUE_QUANT, 0.2 * tol)
    tq = min(_TIME_QUANT, 0.2 * tol / max(max_m, 1.0), min_h / 4.0)
    margin = 0.5 * vq + 0.5 * max_m * tq
    tol_dec = tol - margin  # 抽稀阶段的误差预算

    keep = _thin(curve, tol_dec, src_tan)

    # 切线定型 + 量化后精确验证；违例则插回最差关键帧，直到满足上界
    keep_set = set(keep)
    while True:
        out = _build(curve, keep, src_tan, tq, vq)
        err, _ = _verify(curve, out)
        if err <= tol:
            break
        idx = _worst_insertable_key(curve, out, keep_set)
        if idx is None:
            # 已无可插关键帧：全量保留 + 原切线，误差只剩量化项
            out = Curve(list(times), list(values), curve.cyclic)
            out.tangents = src_tan
            _quantize(out, tq, vq)
            err, _ = _verify(curve, out)
            break
        keep.insert(bisect.bisect(keep, idx), idx)
        keep_set.add(idx)
    if err > tol:
        # 兜底：放弃量化原样返回（理论上不会发生，误差恒为 0）
        out = Curve(list(times), list(values), curve.cyclic)
        out.tangents = src_tan
    out.time_step = tq
    out.value_step = vq
    return out


# ---------------------------------------------------------------- 落盘

def save(curve, path):
    """落盘"""
    payload = {
        "times": curve.times,
        "values": curve.values,
        "tangents": curve.tangents,
        "cyclic": curve.cyclic,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
