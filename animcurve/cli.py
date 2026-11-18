# -*- coding: utf-8 -*-
import argparse
import json
import animcurve as m


def main():
    ap = argparse.ArgumentParser(description="animcurve")
    ap.add_argument("path")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = json.load(open(a.path, encoding="utf-8"))
    c = m.Curve(d["times"], d["values"], d.get("cyclic", False))
    n = m.compress(c, 0.02)
    print(json.dumps({"before": len(c.times), "after": len(n.times)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
