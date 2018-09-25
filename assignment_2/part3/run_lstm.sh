#!/bin/sh
#PBS -lwalltime=48:00:00
#PBS -lnodes=4:ppn=12
#PBS -lmem=250GB

python $HOME/assignment_2/part3/train.py --txt_file "$HOME/assignment_2/part3/book.txt" --save_file "$HOME/assignment_2/part3/model.pt" --summary_path "$HOME/assignment_2/part3/summaries/" --device 'cuda:0' --print_every 1000 --sample_every 1000 &> $HOME/assignment_2/part3/final.txt