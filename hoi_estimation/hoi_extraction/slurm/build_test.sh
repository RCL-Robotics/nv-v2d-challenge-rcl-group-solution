#!/bin/bash
#SBATCH --job-name=v2d-sing-test
#SBATCH --output=/l/users/han.zhou/v2d/logs/%x-%j.out
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail
export SINGULARITY_CACHEDIR=/l/users/han.zhou/v2d/build/cache
export SINGULARITY_TMPDIR=/l/users/han.zhou/v2d/build/tmp
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"
cd /l/users/han.zhou/v2d/build

singularity build --fakeroot mini.sif mini.def
singularity exec mini.sif bash -c "echo FAKEROOT_BUILD_OK && cat /etc/alpine-release"
