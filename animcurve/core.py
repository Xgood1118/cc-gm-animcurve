# -*- coding: utf-8 -*-
"""animcurve"""

from __future__ import annotations

__all__ = ["Curve", "compress", "sample", "save"]


class Curve:
    """Curve"""

    def __init__(self, times, values, cyclic=False):
        """动画曲线"""
        raise NotImplementedError("Curve.__init__ 尚未实现")


def compress(curve, tol):
    """在误差上界内抽稀关键帧"""
    raise NotImplementedError("compress 尚未实现")


def sample(curve, t):
    """按时间采样"""
    raise NotImplementedError("sample 尚未实现")


def save(curve, path):
    """落盘"""
    raise NotImplementedError("save 尚未实现")
