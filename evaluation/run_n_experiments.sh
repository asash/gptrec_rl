N=100
for i in `seq 1 $N`;
do 
	unbuffer python3 run_experiment.py mlp_config_with_lambdarank.py ./results/experiment_$i.json > ./logs/stdout 2> ./logs/stderr;
done;
