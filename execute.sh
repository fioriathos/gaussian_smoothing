#!bin/bash
#File to be analyzed
file='path_of_file'
#Variable we want to do the inference
var='var_to_analyze'
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
source activatepython.sh
#from csv to matrix
python create_mat.py $file $var
echo 'matrix created'
# inference of parameters to be done with array job
cat inferences.sh | sed "s+numarray+$numarray+g" > runinference.sh
echo 'run hyperparam optimization'
sbatch --wait runinference.sh
#print all results in one file
cat hypinfer_* > allhypinfer.txt
## find the best hyperparameters set
python readbestinf.py allhypinfer.txt
echo 'best parameter in allhypinfer.txt found'
# build submatrix to do ARRAY jobs
python dividematrix.py $numproc
#run path prediciton
touch listofjobs.txt
for k in $(ls subnromalized*);do
    cat pathprediction.sh | sed "s+submat+$k+g;s+step+$step+g;s+dt+$dt+g"> runpathprediction.sh
    sbatch runpathprediction.sh >> listofjobs.txt
done
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
echo "DONE! You can find the file $var _ATHOS.csv with all the data"
