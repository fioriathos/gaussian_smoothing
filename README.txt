###########################
#########INSTALL##########
###########################
# Download git repo
git clone https://github.com/fioriathos/gaussian_smoothing.git
# enter folder
cd gaussian_smoothing
# Create python virtual env (this command will create a virtual environment folder ~/gaussian_smoothing/virtualenv/)
# On the cluster I use Python/3.5.2-goolf-1.7.20
python -m venv virtualenv
# Activate virtual environemnt
source virtualenv/bin/activate
# Install all required packages
pip install -r requirements.txt
# load your virtual environment locally
go into gaussian_smooting/execute.sh and change
source /scicore/../activatepython.sh with YOUR virtualenv e.g. 
source ~/gaussian_smoothing/virtualenvironment/bin/activate

- DEPENDING IF YOU WORK LOCALLY OR USING THE CLUSTER YOU HAVE TO CHANGE THE
  execute.sh FILE

###########################
# IF YOU USE THE CLUSTER
##########################
- go to gaussian_smoothing/inference.sh and change "source
  /scicore/home/../activatepython.sh" to the directory of YOUR python virutal
  environement activation e.g. "source
  ~/gaussian_smoothing/virtualenvironment/bin/activate"
- go to gaussian_smoothing/pathprediction.sh and change "source
/scicore/../activatepython.sh" to YOUR virtualenvironment
"source ~/gaussian_smoothing/virtualenvironment/bin/activate"
###########################
# IF YOU DO NOT USE THE CLUSTER
##########################
If you run this in local (deprecated) you should 
-1) CANCEL/COMMENT THE LINES
cat inferences.sh | sed "s+numarray+$numarray+g" | sed "s+dt_a+$dt_a+g" >
runinference.sh
echo 'run hyperparam optimization'
sbatch --wait runinference.sh
cat hypinfer_* > allhypinfer.txt
- and SOBSTITUTE/UNCOMMENT them with
touch allhypinfer.txt
python inference.py $dt_a>>allhypinfer.txt
-2) cancel all the part between ##@@@ for the path prediction and uncomment the
one between #$$$

The file to execute is execute.sh where you have to specify
-The path of the file to analyse
-the variable we are interested in (gfp_nb,length_um,length_vtmvb,concentration,..)
-dt_a the acquisition time (usually 3 min)
# The following you can usually keep the default one
-numarray is the number of time we parallely optimize the likelihood by
    starting from differents initial conditions [higer it is slower will be but
    potentially more precise]
-numproc is the number of times we divide the initial matrix in sub matrices
    [higher it is faster will be] rule of tumb: (total number of
    cells/numproc~10
-step is the time step for the prediction. The input equals 3 [min] but we can
    have prediction every step [min]
-dt is the time step [min] for computing the derivative (x[t+dt]-x[-dt])/2dt 
NB it might be necessary to modify pathprediction.sh or inferences.sh depending
on how much time/memory/.. is necessary for the current jobs running
NB the variable length_vbmvt = vetical_bottom-vertical_top has been created
THE OUTPUT
the only outputfile will be
-var_ATHOS.csv where
    -cell: is the unique identifier of a cell as in original file
    -d_var_dt: the derivative of the variable of interest
    -var_pred: the prediction from GPy
    -err: the variance on the prediction
    -var_raw: the original input variable [every 3 min for MoMa]
    -time_se: the new time step with the correct starting time
-allhyperinfer.txt
    the file containing all the value of the minimization

