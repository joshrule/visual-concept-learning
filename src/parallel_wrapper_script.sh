# This is the wrapper to call to actually run new evaluations that have been
# configured using evaluateFeatureSets.m.

for i in `seq 1 $3`;
do
    ./parallel_"$1"_script.sh $i > ./KR"$1"_$2.$i;
done
