import os 
import sys
from pathlib import Path
import time 
import shutil
checkpoints_dir = Path(sys.argv[1])
print(f"monitoring checkpoints in {checkpoints_dir}")

while True:
    val_checkpoint_steps = []
    val_fname = checkpoints_dir/"validations.csv"
    if not os.path.isfile(val_fname):
        print("no validation file found")
        time.sleep(1)
        continue

    with open(val_fname) as f:
        val_checkpoint_steps = list([int(line.split(',')[0].split('/')[-1].split("_")[-1]) for line in f.readlines()])

    val_checkpoint_steps.sort()   
    if len(val_checkpoint_steps) == 0:
        print("no val checkpoints found")
        time.sleep(1)
        continue

    print(val_checkpoint_steps)

    for fname in os.listdir(checkpoints_dir):
        if fname.startswith("checkpoint_step_"):
            step = int(fname.split("_")[-1])
            if step not in val_checkpoint_steps and step < val_checkpoint_steps[-1]:
                checkpoint_path = checkpoints_dir / fname 
                print(f"deleting {checkpoint_path}")
                shutil.rmtree(checkpoint_path)
                #delete
                pass
            else:
                print(f"keeping step {step}")
    time.sleep(1)
