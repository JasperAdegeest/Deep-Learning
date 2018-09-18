#!/bin/sh
#PBS -lwalltime=00:50:00
#PBS -lnodes=4:ppn=12
#PBS -lmem=250GB

cp -r $HOME/assignment_1/code/cifar10 "$TMPDIR"

python $HOME/assignment_1/code/train_convnet_pytorch.py --data_dir "$TMPDIR/cifar10/cifar-10-batches-py" &> $HOME/assignment_1/code/results_file.txt