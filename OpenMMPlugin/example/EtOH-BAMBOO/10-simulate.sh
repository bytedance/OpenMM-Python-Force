#!/bin/bash
set -xe

# # minimize
# python3 md.py --job minimize --ifile alcohol.sdf --ofile min --mode omm


# numerical tests

pairs1=(
    "nve1 omm"
    "nve2 native"
    "nve3 script"
    "nve4 compile"
    "nve5 omm.graph"
    "nve6 native.graph"
    "nve7 script.graph"
    "nve8 compile.graph"
)
for pair in "${pairs1[@]}"; do
    set -- $pair
    x=$1
    y=$2

    python3 md.py --job simulate --ifile alcohol.sdf min.pdb --nsave 1 --nframe 100 --friction 0 \
        --ofile $x --mode $y
done


# speed tests

pairs2=(
    "d1 omm"
    "d2 native"
    "d3 script"
    "d4 compile"
    "d5 omm.graph"
    "d6 native.graph"
    "d7 script.graph"
    "d8 compile.graph"
)
for pair in "${pairs2[@]}"; do
    set -- $pair
    x=$1
    y=$2

    python3 md.py --job simulate --ifile alcohol.sdf min.pdb --nsave 100 --nframe 10 \
        --ofile $x --mode $y
done
