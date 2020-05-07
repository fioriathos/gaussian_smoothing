#!/bin/bash

#SBATCH --job-name=integratedtocluster                   #This is the name of your job
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


#ADD YOUR VIRTUAL ENVIRONMENT
#################################
source ENV 
python pathprediction.py submat step 
#export your required environment variables below
#################################################


#add your command lines below
#############################



