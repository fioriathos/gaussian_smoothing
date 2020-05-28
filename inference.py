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
#free = {'variance':init[0],'gstds':init[1],'lengthscale':init[2]}
free = {'variance':init[0],'lengthscale':init[2],'gstds':1.5**2/var}
#the parameters we take fixed
# NB! Data are normalized so parameters must be rescaled!
O = np.load('original.npy')
var = np.nanstd(O+1e-08)
fixed = {}
#############################################################
#############################################################
X = np.load('normalized.npy')
T = giveT(X,np.load('dt.npy'))
mm = minrbf.minimize_rbf(time=T,path=X,free=free,fixed=fixed)
if free:
    minimiz = mm.minimize()
    print(minimiz[0])
else:
    np.save('parameters.npy',np.array([fixed['lengthscale'],fixed['variance'],fixed['gstds']]))
