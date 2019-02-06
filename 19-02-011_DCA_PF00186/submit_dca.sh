#!/bin/bash
#===============================================================================
swarm --gres=lscratch:10 -g 64 -p 2 --time 24:00:00 -f dca.swarm

#swarm -f dca.swarm -g 64 -p 2 --time 24:00:00 --module matlab/2018a --logdir log --gres=lscratch:10
