# run python
script='/data1/josh/concept_learning/src/topLevelScripts/new_simulations/evaluatePerformance.py'
csv_file='/data1/josh/concept_learning/evaluation/v0_2/binary_evaluation_input_output_files.csv'

input_file=$(awk -F, -v "line=$1" 'NR==line {print $1}' $csv_file)
output_file=$(awk -F, -v "line=$1" 'NR==line {print $2}' $csv_file)
echo input file: $input_file
echo output file: $output_file

python $script $input_file $output_file
echo END OF JOB
exit 0
