#!/bin/bash
#$ -cwd
#$ -j y 
#$ -S /bin/bash 
#$ -V 
#$ -N CLEval 
#$ -pe smp 32

# I don't remember why it's important to limit the number of memory arenas.
export MALLOC_ARENA_MAX=1

# Initialize key variables.
script='/home/share/data/visual-concept-learning/src/topLevelScripts/new_simulations/evaluatePerformance.py'
csv_file='/home/share/data/visual-concept-learning/evaluation/v0_2/binary_evaluation_input_output_files.csv'
input_file=$(awk -F, -v "line=$SGE_TASK_ID" 'NR==line {print $1}' $csv_file)
output_file=$(awk -F, -v "line=$SGE_TASK_ID" 'NR==line {print $2}' $csv_file)
echo input file: $input_file
echo output file: $output_file
# Activate the virtualenv.
source /home/share/data/visual-concept-learning/binary_evaluation_env/bin/activate
# Run diagnostic tests.
echo
which pip
echo
pip list --format=columns
echo
# Run the script.
python $script $input_file $output_file
# Announce your finish.
echo END OF JOB
exit 0

#### qsub bash call
#qsub -t 1:<lines_in_csv_file> qsub_simulation.sh
