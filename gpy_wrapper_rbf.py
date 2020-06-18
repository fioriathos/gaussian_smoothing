import numpy as np
from numba import jit,float64,vectorize
import GPy
from pathos.multiprocessing import ProcessingPool as Pool
class gpy_wrapper_rbf(object):
    def __init__(self, lengthscale, variance, gstds):
        self.lengthscale= lengthscale 
        self.variance= variance
        self.gstds = gstds
        if type(gstds) is not np.float64:
            self.gstds = gstds[~np.isnan(gstds)][:,None]
    def row_slice(self, xt, nproc):
        """Return sliced array in nproc times"""
        cs = xt.shape[0]//nproc #chuncksize
        tmp = [xt[i*cs:cs*i+cs,:] for i in range(nproc)]
        if nproc*cs != xt.shape[0]:
            tmp[-1] = np.concatenate((tmp[-1],xt[nproc*cs:xt.shape[0],:]),axis=0)
        return tmp
    def create_model(self,t,y):
        """Do OU model """
        assert t.shape[1]==1
        assert y.shape[1]==1
        t=t[~np.isnan(t)][:,None];y=y[~np.isnan(y)][:,None]
        ker = GPy.kern.RBF(1, variance=self.variance, lengthscale=self.lengthscale)
        ker = ker + GPy.kern.Bias(input_dim=1)
        if self.gstds.ndim>1:
            #print ("here", type(gstds),gstds)
            assert t.shape==self.gstds.shape,"control gstds shape"
        m = GPy.models.GPRegression(t, y, ker, noise_var=self.gstds)
        return m
    def give_objective(self,t,path):
        """Take 2 vectors and return a np array with gradient and objective function"""
        #Some test for shape
        if t.ndim==1: t=t[:,None];
        if path.ndim==1: path=path[:,None]
        assert t.shape[1]==1,"tshape";assert path.shape[1]==1,"pshape"
        #Create the model
        mod= self.create_model(t=t,y=path)
        dlen= mod.kern.rbf.lengthscale.gradient_full
        dvar= mod.kern.rbf.variance.gradient_full
        dg_stdS = mod.Gaussian_noise.gradient_full
        dg = -np.array((dlen,dvar,dg_stdS))# -1 since gradient of objective
        return np.append(dg,mod.objective_function())
    def vectorize_objective(self,foo):
        """Allows to apply give_obective on a matrix row by row"""
        t,path,fun = foo
        tmp = lambda x,y: self.give_objective(x,y)
        fun_obj = fun(tmp,signature='(m),(n)->(4)') # means go row by row and return a 5 dimensional vec
        return fun_obj(t,path)
    def tot_grad_obj(self,t,path):
        """return objective (-LL) and toal gradient (dmu,dgam,dsigS,dg_stdS) """
        tmp = self.vectorize_objective([t,path,np.vectorize])
        tmp_add = np.add.reduce(tmp,axis=0)
        #print "LL",tmp_add[-1], "grad",tmp_add[:-1]
        return tmp_add[-1], tmp_add[:-1]
##    def predict(self,t,y,lamb1=None,lamb2=None,time_size=0.5):
#        """Do a prediction with time_size[min]"""
#        ## TO CHECK ##
#        if t.ndim==1: t=t[:,None]
#        if y.ndim==1: y=y[:,None]
#        assert t.shape[1]==1
#        assert y.shape[1]==1
#        assert t.shape==y.shape
#        if lamb1 is not None: assert lamb1.shape[0]==1
#        t=t[~np.isnan(t)][:,None];y=y[~np.isnan(y)][:,None]
#        time = np.arange(t[0],t[-1],time_size)[:,None]
#        m,_ = self.create_model(t,y,normalized=False,lamb1=lamb1,lamb2=lamb2)
#        #optimize initial condition
#        m.constrain_fixed()
#        if self.proc==1: m.mean.l0.unconstrain()
#        else: m.mean.x0.unconstrain()
#        m.optimize()
#        #predict
#        y_pred , err = m.predict_noiseless(time)
#        return time, y_pred,err[0][0][:,None]
#    def loop_predict(self,x):
#        """Apply on multidimensional array"""
#        if self.process== 'integrated_ornstein_uhlenbeck_with_bleaching':
#            times,paths,lamb1,lamb2,dt = x
#            return [self.predict(times[i,:],paths[i,:],\
#                                 lamb1[i,:],lamb2[i,:],time_size=dt) for i in range(times.shape[0])]
#        else:
#            times,paths,dt = x
#            return [self.predict(times[i,:],paths[i,:],time_size=dt) for i in range(times.shape[0])]
#    def parallel_predict(self,t,path,lamb1=None,lamb2=None,nproc=8,dt=0.5):
#        """Allows to apply give_obective on a matrix row by row"""
#        if lamb1 is not None: assert lamb1.shape[0]==t.shape[0]
#        if lamb1 is not None: assert lamb2 is not None
#        pool = Pool(processes=nproc)
#        times = self.row_slice(t,nproc)
#        paths = self.row_slice(path,nproc)
#        if self.process== 'integrated_ornstein_uhlenbeck_with_bleaching':
#            assert lamb1 is not None
#            assert lamb2 is not None
#            lamb1s = self.row_slice(lamb1,nproc)
#            lamb2s = self.row_slice(lamb2,nproc)
#            output =[ pool.amap(self.loop_predict,[(ti,pa,la1,la2,dt)]) for ti,pa,la1,la2 in zip(times,paths,lamb1s,lamb2s)]
#        else:
#            output =[ pool.amap(self.loop_predict,[(ti,pa,dt)]) for ti,pa in zip(times,paths)]
#        outputs = [o.get() for o in output]
#        def concat(a,nmax=t.shape[1]/float(dt)*(t[0,2]-t[0,1])):
#            "Add with nan the missing elements"
#            return np.append(a.T,np.zeros((int(nmax)-a.shape[0],1))*np.nan)
#        outputs = np.array([ map(concat,[x[0],x[1],x[2]]) for j\
#                  in outputs for q in j for x in q  ])
#        itime=outputs[:,0,:]
#        ipath = outputs[:,1,:]
#        ierr = outputs[:,2,:]
#        assert itime.shape[0]==t.shape[0]
#        return itime,ipath,ierr
##################################################################
## Hessian implementation for giving error bars
#################################################################
#    @jit
#    def d2L_dK(self, Kinv, alpha, dK_dth1, dK_dth2, d2K_dth12):
#        #Seems to work properly
#        """Give second order derivative w.r.t. hypeparam theta1 and theata2"""
#        #print "alpha",alpha
#        #print "Kinv",Kinv
#        D = np.linalg.multi_dot([dK_dth2,Kinv,dK_dth1])
#        assert D.shape == d2K_dth12.shape
#        foo = np.dot(alpha.T,D+D.T-d2K_dth12)
#        term1 = np.dot(foo,alpha)
#        term2 = -D+d2K_dth12
#        term2 = np.matmul(Kinv,term2)
#        term2 = np.trace(term2)
#        return -0.5*(term1+term2)
#    @jit
#    def d2L_dm(self, Kinv, alpha, dm_dth1, dm_dth2, d2m_dth12):
#        """Second order der for the mean"""
#        term1 = np.dot(alpha.T,d2m_dth12)
#        term2 = np.dot(Kinv,dm_dth1)
#        term2 = np.dot(dm_dth2.T,term2)
#        return term1-term2
#    @jit
#    def d2L_dKdm(self, Kinv, alpha, dK_dth1, dm_dth2):
#        """Mixed term in mean and kern"""
#        tmp = np.dot(dK_dth1,alpha)
#        tmp = np.dot(Kinv,tmp)
#        tmp = np.dot(dm_dth2.T,tmp)
#        tmp1 = np.dot(Kinv,dm_dth2)
#        tmp1 = np.dot(dK_dth1,tmp1)
#        tmp1 = np.dot(alpha.T,tmp1)
#        return -0.5*(tmp+tmp1)
#    def log_marginal(self,t,y,normalized=False, lamb1=None,lamb2=None):
#        """ret log marginal"""
#        #if normalized:
#        #    raise NotImplementedError
#        assert t.shape[1]==1
#        assert y.shape[1]==1
#        if self.process == 'integrated_ornstein_uhlenbeck_with_bleaching':
#            assert len(lamb1)==len(lamb2)==1,"lamb size"
#        #if there are nan we must get rid of them
#        t=t[~np.isnan(t)][:,None];y=y[~np.isnan(y)][:,None]
#        if normalized:\
#            y,mu,sigmas,gstds,mv,mc=self.normlize(y,lamb1=lamb1,lamb2=lamb2)
#        else: mu,sigmas,gstds,mv,mc=self.non_normalize()
#        means = [GPy.mappings.ornstein_uhlenbeck,GPy.mappings.integrated_ornstein_uhlenbeck,lambda x,y,mu: GPy.mappings.integrated_ornstein_uhlenbeck_with_bleaching(input_dim=x,output_dim=y,mu=mu,lamb1=lamb1,lamb2=lamb2)]
#        kerns = [GPy.kern.ornstein_uhlenbeck,GPy.kern.integrated_ornstein_uhlenbeck,lambda x, variance, gamma: GPy.kern.integrated_ornstein_uhlenbeck_with_bleaching(x,variance,gamma,lamb1=lamb1,lamb2=lamb2)]
#        meanfun = means[self.proc](1,1,mu=mu)
#        cor_ker = kerns[self.proc](1, variance = sigmas, gamma = self.gamma)
#        likelihood = GPy.likelihoods.gaussian.Gaussian(variance=gstds)
#        tmp = GPy.inference.latent_function_inference.\
#                exact_gaussian_inference.ExactGaussianInference().inference\
#                (kern=cor_ker,X=t,Y=y,likelihood=likelihood,\
#                mean_function=meanfun,onlypar=2)
#        return tmp
#    def give_hessian(self,t,y,normalized=False, lamb1=None,lamb2=None):
#        """Return the total hessian for Gaussian likelihood model"""
#        assert t.shape[1]==1
#        assert y.shape[1]==1
#        if self.process == 'integrated_ornstein_uhlenbeck_with_bleaching':
#            assert len(lamb1)==len(lamb2)==1,"lamb size"
#        #if there are nan we must get rid of them
#        t=t[~np.isnan(t)][:,None];y=y[~np.isnan(y)][:,None]
#        if normalized:
#            y,mu,sigmas,gstds,mv,mc=self.normlize(y,lamb1=lamb1,lamb2=lamb2)
#        else: mu,sigmas,gstds,mv,mc=self.non_normalize()
#        means = [GPy.mappings.ornstein_uhlenbeck,GPy.mappings.integrated_ornstein_uhlenbeck,lambda x,y,mu: GPy.mappings.integrated_ornstein_uhlenbeck_with_bleaching(input_dim=x,output_dim=y,mu=mu,lamb1=lamb1,lamb2=lamb2)]
#        kerns = [GPy.kern.ornstein_uhlenbeck,GPy.kern.integrated_ornstein_uhlenbeck,lambda x, variance, gamma: GPy.kern.integrated_ornstein_uhlenbeck_with_bleaching(x,variance,gamma,lamb1=lamb1,lamb2=lamb2)]
#        meanfun = means[self.proc](1,1,mu=mu)
#        cor_ker = kerns[self.proc](1, variance = sigmas, gamma = self.gamma)
#        likelihood = GPy.likelihoods.gaussian.Gaussian(variance=gstds)
#        tmp = GPy.inference.latent_function_inference.\
#                exact_gaussian_inference.ExactGaussianInference().inference\
#                (kern=cor_ker,X=t,Y=y,likelihood=likelihood,\
#                mean_function=meanfun,onlypar=1)
#        alpha = tmp['dL_dm']
#        Kinv = tmp['Kinv']
#        dL_dK = tmp['dL_dK']
#        #first and second derivative, the non spec are 0.
#        tmp = cor_ker.my_grad(t)
#        dK_dsigs = tmp['dK_dsigs']/mv**2
#        dK_dgam= tmp['dK_dgam']
#        tmp = cor_ker.hessian(t)
#        d2K_dgam = tmp['d2K_dgam']
#        d2K_dsigs = tmp['d2K_dsigs']/mv**4
#        d2K_dsigdgam = tmp['d2K_dsigdgam']/mv**2
#        #print "d2kdsi",d2K_dsigs
#        #print "d2kdgam", d2K_dgam
#        #print "d2kdigamdsi", d2K_dsigdgam
#        tmp = meanfun.my_grad(t)
#        dm_dmu = tmp['dm_dmu']/mv
#        dK_derrs = np.identity(Kinv.shape[0])/mv**2
#        K_hessian = np.zeros((4,4))
##        #variable order always mu,gam,sigs,gerrs
#        K_hessian[0,0]=self.d2L_dm(Kinv,alpha,dm_dmu,dm_dmu,np.zeros_like(dm_dmu))
#        K_hessian[0,1]=self.d2L_dKdm(Kinv,alpha,dK_dgam,dm_dmu)
#        K_hessian[0,2]=self.d2L_dKdm(Kinv,alpha,dK_dsigs,dm_dmu)
#        K_hessian[0,3]=self.d2L_dKdm(Kinv,alpha,dK_derrs,dm_dmu)
##        ######
#        K_hessian[1,0]=K_hessian[0,1]
#        K_hessian[1,1]=self.d2L_dK(Kinv, alpha, dK_dgam, dK_dgam, d2K_dgam)
#        K_hessian[1,2]=self.d2L_dK(Kinv, alpha, dK_dgam, dK_dsigs, d2K_dsigdgam)
#        K_hessian[1,3]=self.d2L_dK(Kinv, alpha, dK_dgam,\
#                                   dK_derrs,np.zeros_like(dK_dgam) )
##        #######
#        K_hessian[2,0]=K_hessian[0,2]
#        K_hessian[2,1]=K_hessian[1,2]
#        K_hessian[2,2]=self.d2L_dK(Kinv, alpha, dK_dsigs, dK_dsigs, d2K_dsigs)
#        K_hessian[2,3]=self.d2L_dK(Kinv, alpha, dK_dsigs, dK_derrs,\
#                                  np.zeros_like(dK_dsigs))
##        #################
#        K_hessian[3,0]=K_hessian[0,3]
#        K_hessian[3,1]=K_hessian[1,3]
#        K_hessian[3,2]=K_hessian[2,3]
#        K_hessian[3,3]=self.d2L_dK(Kinv, alpha, dK_derrs, dK_derrs,\
#                                  np.zeros_like(dK_derrs))
#        #Jacobian = np.zeros((4,1))
#        #Jacobian[0,0] = np.sum(alpha*dm_dmu)
#        #Jacobian[1,0] = np.sum(dL_dK*dK_dgam)
#        #Jacobian[2,0] = np.sum(dL_dK*dK_dsigs)
#        #Jacobian[3,0] = np.sum(dL_dK*dK_derrs)
#        #print "Jac", Jacobian
#        return K_hessian
#    def covariance(self, H):
#        "inverse of -1*hessian"
#        Hinv = np.linalg.inv(-H)
#        assert np.allclose(np.matmul(-H,Hinv),np.eye(H.shape[0]))
#        assert sum(np.linalg.eigvals(Hinv)<0)==0, "No positve defined,\
#        something wrong (probably no minimum)"
#        return Hinv
#    def errorbars(self, cov):
#        """Give the error on hyp estimation knowing cov fun"""
#        err = [np.sqrt(cov[i,i]) for i in range(cov.shape[0])]
#        return {'errmu':err[0],'errgam':err[1],\
#                'errsigs':err[2],'errgstds':err[3]}
#    @jit
#    def hessianwrap(self,x):
#        """Sum over all hessians"""
#        T, Y, norm, L1, L2 = x
#        if L1 is None:
#            L1 = [None]*T.size
#            L1 = np.array(L1)
#            L1 = L1.reshape(T.shape) 
#            L2 = L1
#        H = np.zeros((4,4))
#        for k in range(Y.shape[0]):
#            H += self.give_hessian\
#                    (T[k:k+1,:].T, Y[k:k+1,:].T,\
#                     norm, L1[k:k+1,:].T, L2[k:k+1,:].T)
#
#        return H
#    def multiproc_hessian(self,t,path,nproc=5,lamb1=None,\
#                          lamb2=None,normalized=False):
#        """give hessian of several process using multproc"""
#        assert t.shape[0]>=nproc,'decrese proc'
#        """np.vectorize it's not performat, besically a for loop. Do with several proc"""
#        assert t.shape==path.shape
#        pool = Pool(processes=nproc)
#        times = self.row_slice(t,nproc)
#        paths = self.row_slice(path,nproc)
#        if self.process== 'integrated_ornstein_uhlenbeck_with_bleaching':
#            assert lamb1 is not None, "give lambdas"
#            lamb1s = self.row_slice(lamb1,nproc)
#            lamb2s = self.row_slice(lamb2,nproc)
#            output =[pool.amap(self.hessianwrap,[(ti,pa,normalized,lamb1,lamb2)]) for ti,pa,lamb1,lamb2 in zip(times,paths,lamb1s,lamb2s)]
#        else:
#            output =[pool.amap(self.hessianwrap,[(ti,pa,normalized,None, None)]) for ti,pa in zip(times,paths)]
#        H = [o.get() for o in output]
#        H = np.array(H).reshape((nproc,4,4))
#        H = np.sum(H,axis=0)
#        return H 
