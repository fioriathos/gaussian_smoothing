## read csv file sys.argv[1] and build the respective np matrix of
## sys.argv[2]
import pandas as pd
import numpy as np
def create_nan_array(c):
    """From  array of different lengts create one with same length usign np.nan"""
    ma = max([tmp.shape[1] for tmp in c]) #longest path in the list
    foo = [np.hstack((tmp,np.nan*np.zeros((tmp.shape[0],ma-tmp.shape[1])))) for tmp in c]
    return np.vstack(foo)
    #return [np.concatenate((tmp,np.array([np.nan]*tmp.shape[0]*(ma-tmp.shape[1])).reshape((tmp.shape[0],(ma-tmp.shape[1])))),axis=1)for tmp in c]
def give_nparray(df,x):
    """Return the np.array of quantity x with at each row a new cell"""
    tmp = []
    for k in np.unique(df.cell):
        tmp.append(df.loc[df['cell']==k]['{}'.format(x)].values[None,:])
    return np.vstack(create_nan_array(tmp))
def giveT(X,dt):
    """Give time with same X shape at dt min distances"""
    ## Create a time vector
    T = np.repeat(np.arange(0,dt*X.shape[1],dt)[None,:],X.shape[0],axis=0)
    ##Give tha same NAN structure
    T = X*0+T
    return T
if __name__=='__main__':
    import sys
    df = pd.read_csv(sys.argv[1])
    df = df.groupby('cell').filter(lambda x: x.shape[0]>2) # at least 3 data
    X = give_nparray(df,sys.argv[2])
    T = give_nparray(df,'time_sec')/60**2
    #XN = (X - np.nanmean(X,axis=1,keepdims=True))/(np.nanstd(X+1e-08,axis=1,keepdims=True))
    XN = (X - np.nanmean(X,axis=1,keepdims=True))
    ## Save important files
    df.groupby('cell')['time_sec'].first().to_csv('initial_times.csv',header=True)
    np.save('normalized.npy',XN)
    np.save('original.npy',X)
    np.save('names.npy',give_nparray(df,'cell'))
    np.save('times.npy',T)
