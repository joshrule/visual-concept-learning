#!/bin/bash
#$ -cwd
#$ -j y 
#$ -S /bin/bash 
#$ -V 
#$ -m e -M rsj28@georgetown.edu
#$ -o test.log
#$ -N CLSim 
#$ -pe smp 1

# I don't remember why it's important to limit the number of memory arenas.
export MALLOC_ARENA_MAX=1

# Limit the number of CPUs specifically for this test.
export N_PROCS=1

# Run a simple script to prove that you can run simulations.
python -c "import os; print 'hello world!' ; print 'cpus:', os.environ['N_PROCS']" 
exit 0

#### qsub bash call
#qsub test_script.sh
