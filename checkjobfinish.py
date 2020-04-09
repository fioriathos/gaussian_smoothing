import subprocess
import sys
import time
with open(sys.argv[1]) as f:
    content = f.readlines()
f.close()
content = [ j.strip() for j in content ]
content = [ j.replace("Submitted batch job ","") for j in content ]
wait_time=15
for i in range(12000):#after too long stop
    time.sleep(wait_time)
    # INSERT YOUR USERNAME OF THE CLUSTER INSTEAD OF fiori
    message = subprocess.check_output(["squeue","-u","YOURUSERNAME"]) 
    tmp = [j in str(message) for j in content] #check if job in queue
    if sum(tmp)==0:
        time.sleep(wait_time)
        break
