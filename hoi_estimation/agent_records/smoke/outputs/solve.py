#!/usr/bin/env python3
"""Preserve the released baseline when no evidence supports a pose change."""
import argparse
from pathlib import Path
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    np.random.seed(0)
    source = args.inputs / 'reconstruction.npz'
    if not source.is_file():
        parser.exit(2, f'Cannot produce poses: required input is missing: {source}\n'
                    'No output was written; inventing poses or a frame count would be invalid.\n')
    with np.load(source, allow_pickle=False) as data:
        if 'T_world_mesh' not in data:
            parser.exit(2, 'reconstruction.npz has no T_world_mesh array.\n')
        poses = np.array(data['T_world_mesh'], dtype=np.float64, copy=True)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) == 0:
        parser.exit(2, f'Invalid baseline pose shape: {poses.shape}\n')
    if not np.isfinite(poses).all():
        parser.exit(2, 'Baseline poses contain nonfinite values.\n')
    # No image evidence was available during development: do not introduce
    # unsupported orientation, scale, translation, or smoothing changes.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('wb') as output:
        np.savez(output, T_world_mesh=poses)
    print(f'Preserved {len(poses)} baseline poses unchanged in {args.out}')


if __name__ == '__main__':
    sys.exit(main())
