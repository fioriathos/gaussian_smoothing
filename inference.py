#from normalized data do the inference
import minimize_rbf as minrbf
from create_mat import giveT 
import numpy as np
import sys
np.random.seed()
#############################################################
#####################Might be changed########################
#############################################################
#intialize the parameters
init = np.random.uniform(1e-06,1,3)
#the parameters we want to infer
free = {'variance':init[0],'gstds':init[1],'lengthscale':init[2]}
#the parameters we take fixed
# NB! Data are normalized so parameters must be rescaled!
fixed = {}
#############################################################
#############################################################
X = np.load('normalized.npy')
T = giveT(X,float(sys.argv[1]))
mm = minrbf.minimize_rbf(time=T,path=X,free=free,fixed=fixed)
minimiz = mm.minimize()
print(minimiz[0])
