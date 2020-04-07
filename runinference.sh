#!/bin/bash
#SBATCH --job-name=hyperparaminference    #This is the name of your job
#SBATCH --cpus-per-task=10                  #This is the number of cores reserved
#SBATCH --mem-per-cpu=1G              #This is the memory reserved per core.

#SBATCH --time=00:30:00        #This is the time that your task will run
#SBATCH --qos=30min #You will run in this queue
#SBATCH --output=hypinfer_%a.out
#SBATCH --array=1-10

#This job runs from the current working directory


#Remember:
#The variable $TMPDIR points to the local hard disks in the computing nodes.
#The variable $HOME points to your home directory.
#The variable $JOB_ID stores the ID number of your task.


#load your required modules below
#################################
source /scicore/home/nimwegen/fiori/protein_production/mother_machine_inference_algo/activatepython.sh
## The two lag means shold match at N=5000
python inference.py 
#export your required environment variables below
#################################################


#add your command lines below
#############################



