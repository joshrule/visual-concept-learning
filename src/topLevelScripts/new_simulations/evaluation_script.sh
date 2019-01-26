#!/bin/bash
#$ -cwd
#$ -j y 
#$ -S /bin/bash 
#$ -V 
#$ -N CLEval 
#$ -pe smp 32

export MALLOC_ARENA_MAX=1

# run python
script='/home/share/data/visual-concept-learning/src/topLevelScripts/new_simulations/evaluatePerformance.py'
csv_file='/home/share/data/visual-concept-learning/evaluation/v0_2/binary_evaluation_input_output_files.csv'
input_file=$(awk -F, -v "line=$SGE_TASK_ID" 'NR==line {print $1}' $csv_file)
output_file=$(awk -F, -v "line=$SGE_TASK_ID" 'NR==line {print $2}' $csv_file)
echo input file: $input_file
echo output file: $output_file
source /home/share/data/visual-concept-learning/binary_evaluation_env/bin/activate
echo
which pip
echo
pip list --format=columns
echo
python $script $input_file $output_file
python /home/rsj28/email_test.py
echo END OF JOB
exit 0

#### qsub bash call
#qsub -t 1:<lines_in_csv_file> qsub_simulation.sh
