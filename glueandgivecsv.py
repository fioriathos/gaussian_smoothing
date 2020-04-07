import sys
import numpy as np
import pandas as pd
from create_mat import create_nan_array, giveT
def loadinlist(names):
    """load different npy and return a huge npy"""
    tmp = [np.load(j) for j in names]
    tmp = np.array(create_nan_array(tmp))
    return tmp
    #return tmp.reshape(tmp.shape[0]*tmp.shape[1],tmp.shape[2])
def createname(na,leng):
    return list(map(lambda x: na+x+'.npy',map(str,range(leng))))
def isnan(num):
    return num!=num
if __name__=="__main__":
    nfiles = int(sys.argv[1])
    vn = sys.argv[2] #name of variable
    pwd = sys.argv[3] #pwd of file
    step = float(sys.argv[4]) #time step of prediction
    O = np.load('original.npy')
    N = np.load('names.npy')
    der = loadinlist(createname('D',nfiles))
    path = loadinlist(createname('X',nfiles))
    errpat = loadinlist(createname('XE',nfiles))
    assert O.shape[0]==der.shape[0]
    # return in normal space
    vY = np.nanstd(O+1e-08)
    mY = np.nanmean(O)
    path = path*vY+mY
    errpat = errpat*vY**2
    der = der*vY
    # make every np array same shape
    N = np.repeat(N[:,:1],der.shape[1],axis=1)
    #give same nan structure as der
    for j in np.argwhere(isnan(der)):
        N[j[0],j[1]]=np.nan
    O = np.concatenate((O,np.zeros((O.shape[0],\
                        der.shape[1]-O.shape[1]))*np.nan),axis=1)
    T = giveT(der,step)
   #save into panda frame
    df = pd.DataFrame(data={'cell':N.reshape(1,-1)[0].tolist(),\
                        'time':T.reshape(1,-1)[0].tolist(),\
                        '{}_pred'.format(vn):path.reshape(1,-1)[0].tolist(),\
                        'err':errpat.reshape(1,-1)[0].tolist(),\
                        'd_{}_dt'.format(vn):der.reshape(1,-1)[0].tolist(),\
                        '{}'.format(vn):O.reshape(1,-1)[0].tolist()})
    #dfin = df.dropna()
    df.to_csv('{}_ATHOS.csv'.format(vn))
