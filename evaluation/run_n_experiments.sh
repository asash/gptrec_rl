config=$1
N=$2

config_filename=$(basename -- "$config")
config_id="${config_filename%.*}"



date=`date +%Y_%m_%dT%H_%M_%S`
experiment_id="${config_id}_${date}"
root_dir=./results/$experiment_id
experiment_stdout=$root_dir/stdout
experiment_stderr=$root_dir/stderr

mkdir $root_dir

echo experement resutls are saved at $root_dir

cp $config $root_dir

for i in `seq 1 $N`;
do 
    experiment_result=$root_dir/experiment_${i}.json
    echo "experiment_stdout: ${experiment_stdout}"
    echo "experiment_stderr: ${experiment_stderr}"
    echo "experiment_result: ${experiment_result}"
	unbuffer python3 run_experiment.py $config $experiment_result > $experiment_stdout 2> $experiment_stderr;
done;
