#!/bin/bash
#SBATCH --job-name=hyperparaminference    #This is the name of your job
#SBATCH --cpus-per-task=10                  #This is the number of cores reserved
#SBATCH --mem-per-cpu=5G              #This is the memory reserved per core.

#SBATCH --time=06:00:00        #This is the time that your task will run
#SBATCH --qos=6hours #You will run in this queue
#SBATCH --output=hypinfer_%a.out
#SBATCH --array=1-2

#This job runs from the current working directory


#Remember:
#The variable $TMPDIR points to the local hard disks in the computing nodes.
#The variable $HOME points to your home directory.
#The variable $JOB_ID stores the ID number of your task.


# LOAD YOUR VIRTUAL ENVIRONMENT!
#################################
source /scicore/home/nimwegen/fiori/protein_production/mother_machine_inference_algo/activatepython.sh
## The two lag means shold match at N=5000
python inference.py 3 
#export your required environment variables below
#################################################


#add your command lines below
#############################



