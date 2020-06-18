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
#free = {'variance':init[0],'lengthscale':init[1],'gstds':init[2]}
free = {'variance':init[0],'lengthscale':init[1]}
#the parameters we take fixed
# NB! Data are normalized so parameters must be rescaled!
fixed = {'gstds':(1.5*0.065)**2} # 1.5 pixel times conv squered
#############################################################
#############################################################
X = np.load('normalized.npy')
T = np.load('times.npy')
# Test set subsempled
#print(T.shape[0], int(sys.argv([1])))
idx = np.random.randint(T.shape[0], size=int(sys.argv[1]))
T = T[idx,:];X = X[idx,:]
# Run param pred
mm = minrbf.minimize_rbf(time=T,path=X,free=free,fixed=fixed)
if free:
    minimiz = mm.minimize()
    # Objective per number of points
    minimiz[0]['fun']=minimiz[0]['fun']/np.sum(~np.isnan(T.reshape(-1)))
    print(minimiz[0])
else:
    np.save('parameters.npy',np.array([fixed['lengthscale'],fixed['variance'],fixed['gstds']]))
