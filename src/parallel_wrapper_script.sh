for i in `seq 1 $2`;
do
    ./parallel_evaluation_script.sh $i > ./KREval_$1.$i;
done
