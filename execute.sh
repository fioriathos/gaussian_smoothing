#!bin/bash
#File to be analyzed
file='path_of_file'
#Variable we want to do the inference
var='var_to_analyze'
#Acquisition time in min dt usually 3min
dt_a=3
#Number of times hyperparameters are optimized
#higher this number slower will be but more precise
numarray=10
#Number of separate process to predict paths
# Higer this number faster will be
numproc=11
#Time intereval between predictions [min]
step=1
#Dt for computing the derivative
dt=0.1
############################################
############################################
######## SOBSTITUTE WITH YOUR VIRTUAL ENV
#source /scicore/home/nimwegen/fiori/protein_production/mother_machine_inference_algo/activatepython.sh
source /Users/fiori/PHD/gaussian_smoothing/virtualenv/bin/activate
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
cat inferences.sh | sed "s+numarray+$numarray+g;s+dt_a+$dt_a+g" > runinference.sh
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
#for k in {1..$numarray};do
#    python inference.py $dt_a & >> allhypinfer.txt
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
#@@@
for k in $(ls subnromalized*);do
    cat pathprediction.sh | sed "s+submat+$k+g;s+step+$step+g;s+dt+$dt+g;s+dt_a+$dt_a+g"> runpathprediction.sh
    sbatch runpathprediction.sh >> listofjobs.txt
done
#@@@
################################
##########  LOCAL     ##########
################################

# Lines among simbols #$$$ have to be delete if we work wiht cluster 
#$$$
for k in $(ls subnromalized*);do
    python pathprediction.py $k $step $dt $dt_a >> listofjobs.txt &
done

wait
#$$$
####------------------------------------###
echo 'predict the paths...'
#wait until all jobs are done
python checkjobfinish.py listofjobs.txt
#glue the final csv
echo "almost done.."
python glueandgivecsv.py $numproc $var $file $step
#delete useless file
#rm *.out
#rm listofjobs.txt
#rm *.npy
echo "DONE! You can find the file $var _ATHOS.csv with all the data"
