config=$1
N=$2

experiment_id=`date -Iseconds`
root_dir=./results/$experiment_id

mkdir $root_dir

echo experement resutls are saved at $root_dir

cp $config $root_dir

for i in `seq 1 $N`;
do 
	unbuffer python3 run_experiment.py $config $root_dir/experiment_$i.json > $root_dir/stdout 2> $root_dir/stderr;
done;
