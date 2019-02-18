for i in `seq 1 $3`;
do
    ./parallel_"$1"_script.sh $i > ./KR"$1"_$2.$i;
done
