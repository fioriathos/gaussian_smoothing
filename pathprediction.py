import GPy
import numpy as np
def predict(T,X,param,step,dt):
    #do prediction even for derivative
    T = T[~np.isnan(T)][:,None]
    X = X[~np.isnan(X)][:,None]
    ker =GPy.kern.RBF(1,lengthscale=param['lengthscale'],\
                     variance=param['variance'])
    m = GPy.models.GPRegression(X=T,Y=X,kernel=ker,noise_var=param['gstds'])
    newT = np.arange(min(T),max(T),step)[:,None]
    Tm,Terr = m.predict_noiseless(newT)
    Tpt,_ = m.predict_noiseless(newT+dt)
    Tmt,_ = m.predict_noiseless(newT-dt)
    return Tm, Terr, (Tpt-Tmt)/(2*dt)
if __name__=='__main__':
    import sys
    from create_mat import giveT,create_nan_array
    # file to load
    submat=sys.argv[1]
    # parameters from inference
    par = np.load('parameters.npy')
    param={}
    param['lengthscale']=par[0]
    param['variance']=par[1]
    param['gstds']=par[2]
    #step for the predicitons
    step = float(sys.argv[2])
    #delta tau for derivatives
    dt = float(sys.argv[3])
    #laod submatrix
    X = np.load(submat)
    T = giveT(X)
    der = []; path=[]; errpath=[]
    for k in range(T.shape[0]):
        tm,te,de = predict(T[k:k+1,:],X[k:k+1,:],param,step,dt)
        path.append(tm.T); errpath.append(te.T); der.append(de.T)
    D = np.array(create_nan_array(der))
    X = np.array(create_nan_array(path))
    XE = np.array(create_nan_array(errpath))
    np.save(submat.replace('subnromalized','D'),D.reshape(D.shape[0],D.shape[-1]))
    np.save(submat.replace('subnromalized','X'),X.reshape(D.shape[0],D.shape[-1]))
    np.save(submat.replace('subnromalized','XE'),XE.reshape(D.shape[0],D.shape[-1]))
