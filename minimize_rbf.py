import GPy
from numpy.random import randn
from numba import jit
from pathos.multiprocessing import ProcessingPool as Pool
import numpy as np
import gpy_wrapper_rbf as gpw
class minimize_rbf(object):
    """Do minimization on """
    def __init__(self,time,path,free,fixed={},\
                 method='TNC',boundary=None):
        """These 3 dictionary fixed the free param, constrained param, boundary"""
        assert time.shape == path.shape;assert type(fixed)==dict
        assert type(free)==dict;assert free!={}
        self.fixed = fixed; self.free = free
        self.time = time; self.path = path
        self.cons=None
        self.method = method
        if boundary==None: # Only positive values allowed
            self.boundary = [(1e-12,None)]*len(free)
        self.cons = None
      #moedl amoung OU-OUInt-OUINTBLE
        assert set(fixed.keys())|set((free.keys()))==set(('gstds', 'variance',\
                                                          'lengthscale'))
        def set_att(key,val):
            if key=='lengthscale':self.lengthscale=val
            if key=='variance':self.variance=val
            if key=='gstds':self.gstds=val
        #set attributes
        for key,val in free.items():
            set_att(key,val)
        for key,val in fixed.items():
            set_att(key,val)
    def fix_par(self,vec, **kwargs):
        """From vec divide the fix and not fixed"""
        from collections import OrderedDict
        vecout = {}
        tmp = OrderedDict([ ('lengthscale',vec[0]),( 'variance',vec[1]),( 'gstds',vec[2])])
        for key in kwargs:
            vecout[key]=tmp[key]
            del tmp[key]
        return np.array([tmp[key] for key in tmp]), vecout
    def rebuild_param(self,vec,**kwargs):
        """From vec and a dictionary with fixed rebuild the correct vec"""
        from collections import OrderedDict
        tmp = OrderedDict([('lengthscale',None),( 'variance',None),( 'gstds',None)])
        for key,val in kwargs.items():
            assert val!=None, "Can't have None as fixed values"
            tmp[key]=val
        for key,val in tmp.items():
            if val==None:
                tmp[key]=vec[0]
                vec = np.delete(vec,0)
        return np.array([tmp[key] for key in tmp])
    def tot_objective(self,x):
        lengthscale,variance,gstds=x
        mod = gpw.gpy_wrapper_rbf(lengthscale,variance,gstds)
        ### Modify Nor True
        tmp = mod.tot_grad_obj(self.time,self.path)
        return tmp
    def return_model(self):
        return gpw.gpy_wrapper_rbf(self.lengthscale,self.variance\
                               ,self.gstds)
    def tot_grad_obj(self,x0,gradient):
        """Return total obj and grad depending on the x0 np.array"""
        tmp = self.tot_objective(self.rebuild_param(x0,**self.fixed))
        obj = tmp[0]
        if gradient:
            grad =self.fix_par(tmp[1], **self.fixed)[0]
            return obj,grad
        else:
            return obj
    def initialize(self):
        """Return the x np array"""
        x0 = [None]*3
        for i in self.free:
            if i=='lengthscale':x0[0]=self.free[i]
            if i=='variance':x0[1]=self.free[i]
            if i=='gstds':x0[2]=self.free[i]
        x0 = [x for x in x0 if x is not None]
        return np.array(x0)
    def minimize_both_vers(self,x0=None,numerical=False):
        """Minimize module.tot_grad_obj(t,path) at point x0={mu:,sigmas,..} considering dic['fix]={mu:,..}"""
        from scipy.optimize import minimize
        if x0 is None:
            x0 = self.initialize()
        if numerical:
            tmp = minimize(self.tot_grad_obj, x0, args=(False),\
                           method=self.method,jac = False,\
                           bounds=self.boundary,\
                           constraints = self.cons,\
                           options={'maxiter':max(1000, 10*len(x0))})
        else:
            #print "SOMETIMES PROBLEMS WITH ANALYTICAL GRADIENT"
            tmp = minimize(self.tot_grad_obj, x0, args=(True),\
                           method=self.method,jac = True,\
                           bounds=self.boundary,\
                           constraints = self.cons,\
                           options={'maxiter':max(1000, 10*len(x0))})
        total_par = self.rebuild_param(tmp['x'],**self.fixed)
        lik_grad = self.tot_grad_obj(tmp['x'],gradient=True)
        return tmp,total_par,lik_grad
    def minimize(self,x0=None):
        """Use Analytical gradient until it workds. Then use numerical in case"""
        import time
        start_time = time.time()
        tmp,total_par,lik_grad = self.minimize_both_vers(numerical=False,x0=x0)
        if tmp['success']==False:
            print("Probably a problem with gradient, do numerical")
            tmp,total_par,lik_grad = self.minimize_both_vers(x0=tmp['x'],numerical=True)
        print("--- %s seconds ---" % (time.time() - start_time))
        self.lengthscale = total_par[0]
        self.variance = total_par[1]
        self.gstds = total_par[2]
        tmp['fx']=np.array([total_par[0],total_par[1],total_par[2]])
        return tmp,total_par,lik_grad
    @jit
    def row_slice(self, xt, nproc):
        """Return sliced array in nproc times"""
        if nproc is None: nproc = self.nproc
        cs = xt.shape[0]//nproc #chuncksize
        tmp = [xt[i*cs:cs*i+cs,:] for i in range(nproc)]
        if nproc*cs != xt.shape[0]:
            tmp[-1] = np.concatenate((tmp[-1],xt[nproc*cs:xt.shape[0],:]),axis=0)
        return tmp
#    def minimize_with_restart(self,num_restart,nproc=None):
#        """Do the minimization with num_restart times using normal dist with
#        0.01 CV"""
#        def minimize_wrapper(matx0):
#            return [self.minimize(matx0[i,:]) for i in range(matx0.shape[0])]
#
#        x0 = self.initialize()[:,None]
#        x0s = np.apply_along_axis(lambda x:\
#                               0.01*x*randn()+x,axis=0,\
#                                arr=np.repeat(x0,num_restart,axis=1)).T
#        slx0s = self.row_slice(x0s,nproc)
#        pool = Pool(processes=nproc)
#        outputs = [pool.amap(minimize_wrapper,[x0]) for x0 in slx0s]
#        outputs = [o.get() for o in outputs]
#        return outputs
##    def heat_map_variables(self,step):
##        import math
#        log2 = lambda x: math.log(x, 2.0)
#        ran = lambda x: x[1]*2**(np.arange(-log2(x[0]),log2(x[0])+step,step)) #how to create the sampling for the heat map
#        folds = [self.mufold,self.gammafold,self.sigmasfold,self.stdsfold]
#        #The natural system for the values
#        valus =[self.mu,1./self.gamma ,np.sqrt(self.sigmas)\
#                ,np.sqrt(self.gstds) ]
#        self.mus,self.gammas,self.sigmass,self.gstdss = map(ran,zip(folds,valus))
#        return [(mu,gamma,sigmas,gstds) for mu in self.mus for gamma in\
#                1./self.gammas for sigmas in self.sigmass**2 for gstds in\
##                self.gstdss**2]
#    def heat_map_variables(self,heatmapvar):
#        assert  type(heatmapvar['mu']) == np.ndarray and \
#                type(heatmapvar['sigmas']) == np.ndarray and \
#                type(heatmapvar['gstds']) == np.ndarray and \
#                type(heatmapvar['gamma']) == np.ndarray
#        print( "EASIER TO THINK ABOUT TAU (1/gamma) , SIGMA ADN GSTD (no\
#            square) RANGE FOR A GOOD HEATMAP!!!")
#        return [(mu,gamma,sigmas,gstds) for mu in heatmapvar['mu'] for gamma in\
#                heatmapvar['gamma'] for sigmas in heatmapvar['sigmas']  for gstds in\
#                heatmapvar['gstds']]
#    def heat_map(self,heatmapvar):
#        """Return the heat map"""
#        pool = Pool(processes=self.nproc)
#        var = self.heat_map_variables(heatmapvar)
#        def func(x):
#            return self.tot_objective(x)[0],x
#        output =[ pool.amap(func,[x]) for x in var]
#        outputs = [o.get() for o in output]
#        outputs = [ [x[0],x[1][0],x[1][1],x[1][2],x[1][3]] for j in outputs for x in j ]
#        return np.array(outputs)
#    def errorbars(self, normalized=False):
#        """Return the inverse of the covariance function"""
#        print( "make sure parameters at minimum")
#        print( "mu",self.mu,"gamma",self.gamma,\
#                'sigmas',self.sigmas,'gstds',self.gstds)
#        gpy = self.return_model()
#        H = gpy.multiproc_hessian(self.time,self.path,self.nproc,\
#                                  self.lamb1,self.lamb2,normalized)
#        cov = gpy.covariance(H)
#        err = gpy.errorbars(cov)
#        return {'relmu':err['errmu']/self.mu,\
#                'relgam':err['errgam']/self.gamma,\
#                'relsigmas':err['errsigs']/self.sigmas,\
#                'relgstds':err['errgstds']/self.gstds}
#    def hessian(self,normalized=False):
#        gpy = self.return_model()
#        H = gpy.multiproc_hessian(self.time,self.path,self.nproc,\
#                                  self.lamb1,self.lamb2,normalized)
#        return H 
#    def predict(self,alpha):
#        """predict paths with used alpha scaling"""
#        gpy = self.return_model()
#        #time path error
#        return gpy.parallel_predict(self.time,self.path,self.lamb1,self.lamb2,nproc=8,dt=0.5)
