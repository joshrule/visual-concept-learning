#!/bin/bash
#$ -cwd
#$ -j y 
#$ -S /bin/bash 
#$ -V 
#$ -N CLSim 
#$ -pe smp 32

export MALLOC_ARENA_MAX=1

# load functions and call simulation() 
matlab -nosplash -nodisplay -nodesktop -r "clear; clc; addpath(genpath('/home/share/data/visual_concept_learning/src/')); simulation(linear5050Params()); exit;"
python /home/rsj28/email_test.py
echo END OF JOB
exit 0

#### qsub bash call
#qsub simulation_script.sh
