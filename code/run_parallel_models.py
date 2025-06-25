import subprocess
import sys
import time
import os

def launchTask(script):
  #    print "Launching: ", script
  task = subprocess.Popen(sys.executable+ " " + script, shell=True, executable="/bin/bash")
  return task


# ------------------------- main -----------------------------------------------

spatial_sizes = [5, 9, 15, 21, 31, 47, 63, 81]
bacth_sizes = {64}
POLL_INTERVAL = 15.  # seconds between checking status of tasks
MAX_CONCURRENT = 5

scripts = []
for s in spatial_sizes:
  for b in bacth_sizes:
    scripts.append(f'model_trainer.py --size={s} --batch_size={b}')

# fire off up-to MAX_CONCURRENT subprocesses...
tasks = list()
for i, script in enumerate(scripts):
  if i >= MAX_CONCURRENT:
    break
  tasks.append(launchTask(script))

scripts = scripts[len(tasks):]  # remove those scripts we've just launched...

while len(tasks) > 0:
  finishedList = []
  for task in tasks:
    retCode = task.poll()
    if retCode != None:
      finishedList.append(task)
      # more scripts to be run?
      if len(scripts) > 0:
        tasks.append(launchTask(scripts[0]))
        del scripts[0]
  for task in finishedList:
    tasks.remove(task)

  time.sleep(POLL_INTERVAL)

print("run_parallel_models.py: Done!:")
