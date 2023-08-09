
from multiprocessing.context import SpawnProcess
import os
from pathlib import Path
import shutil
import time


class CheckpointCleaner(object):
    def __init__(self, checkpoints_dir) -> None:
        self.checkpoints_dir = checkpoints_dir

    def __call__(self):
        checkpoints_dir = Path(self.checkpoints_dir)
        while True:
            val_checkpoint_steps = []
            val_fname = checkpoints_dir/"validations.csv"
            if not os.path.isfile(val_fname):
                print("Checkpoint Cleaner: no validation file found")
                time.sleep(1)
                continue

            with open(val_fname) as f:
                val_checkpoint_steps = list([int(line.split(',')[0].split('/')[-1].split("_")[-1]) for line in f.readlines()])

            val_checkpoint_steps.sort()
            if len(val_checkpoint_steps) == 0:
                print("Checkpoint Cleaner: no val checkpoints found")
                time.sleep(1)
                continue

            for fname in os.listdir(checkpoints_dir):
                if fname.startswith("checkpoint_step_"):
                    step = int(fname.split("_")[-1])
                    if step not in val_checkpoint_steps and step < val_checkpoint_steps[-1]:
                        checkpoint_path = checkpoints_dir / fname
                        print(f"deleting {checkpoint_path}")
                        shutil.rmtree(checkpoint_path)
                    else:
                        pass
            time.sleep(5)

class CheckpointsCleanerProcess(object):
    def __init__(self, checkpoints_dir):
        self.checkpoint_cleaner = CheckpointCleaner(checkpoints_dir)

    def __enter__(self):
        self.validator_process = SpawnProcess(target=self.checkpoint_cleaner)
        self.validator_process.daemon = True
        self.validator_process.start()
        return self
            

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.validator_process.terminate()
        self.validator_process.join()

