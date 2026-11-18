# -*- coding: utf-8 -*-
import argparse
import json

import animcurve as m
from animcurve.core import max_deviation


def main():
    ap = argparse.ArgumentParser(description="animcurve")
    ap.add_argument("path")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tol", type=float, default=0.02)
    a = ap.parse_args()
    with open(a.path, encoding="utf-8") as f:
        d = json.load(f)
    c = m.Curve(d["times"], d["values"], d.get("cyclic", False))
    n = m.compress(c, a.tol)
    stats = {
        "before": len(c.times),
        "after": len(n.times),
        "max_dev": max_deviation(c, n),
        "ratio": len(n.times) / len(c.times),
    }
    if a.out:
        m.save(n, a.out)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
