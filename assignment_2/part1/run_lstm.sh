#!/bin/sh
#PBS -lwalltime=00:50:00
#PBS -lnodes=4:ppn=12
#PBS -lmem=250GB

python $HOME/assignment_2/part1/train.py