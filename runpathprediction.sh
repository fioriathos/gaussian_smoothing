#!/bin/bash

#SBATCH --job-name=integrateocluster                   #This is the name of your job
#SBATCH --cpus-per-task=5                 #This is the number of cores reserved
#SBATCH --mem-per-cpu=1G              #This is the memory reserved per core.

#SBATCH --time=00:30:00        #This is the time that your task will run
#SBATCH --qos=30min #You will run in this queue

# Paths to STDOUT or STDERR files should be absolute or relative to current working directory
#SBATCH

#This job runs from the current working directory


#Remember:
#The variable $TMPDIR points to the local hard disks in the computing nodes.
#The variable $HOME points to your home directory.
#The variable $JOB_ID stores the ID number of your task.


#load your required modules below
#################################
source /scicore/home/nimwegen/fiori/protein_production/mother_machine_inference_algo/activatepython.sh
python pathprediction.py subnromalized9.npy  
#export your required environment variables below
#################################################


#add your command lines below
#############################



