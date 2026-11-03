#!/usr/bin/env python3
"""Compare two CoCoNet bundles (.pth) or two .npz files leaf by leaf; prints max / mean |diff| of every leaf that
differs and the number of identical leaves. Run it in the CARI4D image (the bundles need its torch to unpickle).
Usage: compare_bundles.py <a> <b>"""
import sys

import numpy as np
import torch


def leaves(x, p=""):
    if isinstance(x, dict):
        for k in x:
            yield from leaves(x[k], f"{p}/{k}")
    elif isinstance(x, (list, tuple)) and x and not isinstance(x[0], (int, float, str)):
        for i, v in enumerate(x):
            yield from leaves(v, f"{p}[{i}]")
    else:
        yield p, x


def arr(v):
    if torch.is_tensor(v):
        return v.detach().cpu().double().numpy() if v.is_floating_point() else v.cpu().numpy()
    try:
        return np.asarray(v)
    except Exception:
        return None


def load(f):
    if f.endswith(".npz"):
        z = np.load(f, allow_pickle=True)
        return {k: z[k] for k in z.files}
    return torch.load(f, map_location="cpu", weights_only=False)


a, b = load(sys.argv[1]), load(sys.argv[2])
A, B = dict(leaves(a)), dict(leaves(b))
same = 0
for k in sorted(set(A) | set(B)):
    if k not in A or k not in B:
        print("missing", k)
        continue
    x, y = arr(A[k]), arr(B[k])
    if x is None or y is None or x.dtype == object:
        if str(A[k]) != str(B[k]):
            print("object differs", k)
        else:
            same += 1
        continue
    if x.shape != y.shape:
        print("shape", k, x.shape, y.shape)
        continue
    if np.array_equal(x, y):
        same += 1
        continue
    d = np.abs(x.astype(np.float64) - y.astype(np.float64))
    print(f"{k} {x.shape} max|d| {d.max():.3g} mean {d.mean():.3g} n_diff {(d > 0).sum()}/{d.size}")
print("identical leaves:", same)
