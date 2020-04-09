#!bin/bash
#File to be analyzed
file='file_to_analyse'
#Variable we want to do the inference
var='var_to_analyse'
#Acquisition time in min dt usually 3min
dt_a=3
#Number of times hyperparameters are optimized
#higher this number slower will be but more precise
numarray=10
#Number of separate process to predict paths
# Higer this number faster will be
numproc=11
#Time intereval between predictions [min]
step=$dt_a
#Dt for computing the derivative
dt=0.1
#Python virutal environment
env='/scicore/home/nimwegen/fiori/protein_production/mother_machine_inference_algo/activatepython.sh'
#Scicore uname
sciu='fiori'
#Generate file with correct uname
cat checkjobfinish.txt | sed "s+SCIU+$sciu+g">checkjobfinish.py
############################################
############################################
######## SOBSTITUTE WITH YOUR VIRTUAL ENV
source $env
#from csv to matrix
python create_mat.py $file $var
echo 'matrix created'
####------------------------------------###
################################
########## ON CLUSTER ##########
################################
# inference of parameters to be done with array job
# Lines among simbols #@@@ have to be delete if we work locally
#@@@
cat inferences.sh | sed "s+numarray+$numarray+g;s+ENV+$env+g;s+dt_a+$dt_a+g" > runinference.sh
echo 'run hyperparam optimization'
sbatch --wait runinference.sh
#print all results in one file
cat hypinfer_* > allhypinfer.txt
#@@@
################################
##########    LOCAL   ##########
################################
# Lines among simbols #$$$ have to be delete if we work locally
#$$$
#touch  allhypinfer.txt
#for k in {1..10};do
#    python inference.py $dt_a >> allhypinfer.txt 2>&1 & 
#done
#wait
#$$$

####------------------------------------###


## find the best hyperparameters set
python readbestinf.py allhypinfer.txt
echo 'best parameter in allhypinfer.txt found'
# build submatrix to do ARRAY jobs
python dividematrix.py $numproc
#run path prediciton
touch listofjobs.txt
####------------------------------------###
################################
##########  ON  CLUSTER ##########
################################
# Lines among simbols #@@@ have to be delete if we work locally

##@@@
for k in $(ls subnromalized*);do
    cat pathprediction.sh | sed "s+submat+$k+g;s+ENV+$env+g;s+step+$step+g;s+dt_a+$dt_a+g;s+dt+$dt+g"> runpathprediction.sh
    sbatch runpathprediction.sh >> listofjobs.txt
done
##@@@

################################
##########  LOCAL     ##########
################################

# Lines among simbols #$$$ have to be delete if we work wiht cluster 

##$$$

#for k in $(ls subnromalized*);do
#    python pathprediction.py $k $step $dt $dt_a >> listofjobs.txt 2>&1 &
#done
#wait

#$$$

####------------------------------------###
echo 'predict the paths...'
#wait until all jobs are done
python checkjobfinish.py listofjobs.txt
#glue the final csv
echo "almost done.."
python glueandgivecsv.py $numproc $var $file $step
#delete useless file
rm *.out
rm listofjobs.txt
rm *.npy
rm initial_times.csv
echo "DONE! You can find the file $var _ATHOS.csv with all the data"
