#!/bin/bash
#$ -cwd
#$ -j y 
#$ -S /bin/bash 
#$ -V 
#$ -m e -M rsj28@georgetown.edu
#$ -o test.log
#$ -N CLSim 
#$ -pe smp 1

# no modules on compute nodes
# module load matlab/R2017b
export MALLOC_ARENA_MAX=1
export N_PROCS=1

# load functions and call simulation() 
python -c "import os; print 'hello world!' ; print 'cpus:', os.environ['N_PROCS']" 
exit 0

#### qsub bash call
#qsub test_script.sh
