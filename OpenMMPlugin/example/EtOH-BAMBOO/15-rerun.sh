#!/bin/bash
set -xe

baseline=nve1.pdb
pairs1=(
    "rerun1.npz omm"
    "rerun2.npz native"
    "rerun3.npz script"
    "rerun4.npz compile"
    "rerun5.npz omm.graph"
    "rerun6.npz native.graph"
    "rerun7.npz script.graph"
    "rerun8.npz compile.graph"
)
for pair in "${pairs1[@]}"; do
    set -- $pair
    x=$1
    y=$2

    python3 md.py --job analyze --ifile alcohol.sdf $baseline --friction 0.0 \
        --ofile $x --mode $y
done
