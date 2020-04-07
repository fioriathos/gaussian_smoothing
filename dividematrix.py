import numpy as np
def row_slice( xt, nproc):
    """Return sliced array in nproc times"""
    cs = xt.shape[0]//nproc #chuncksize
    tmp = [xt[i*cs:cs*i+cs,:] for i in range(nproc)]
    if nproc*cs != xt.shape[0]:
        tmp[-1] = np.concatenate((tmp[-1],xt[nproc*cs:xt.shape[0],:]),axis=0)
    return tmp
if __name__=='__main__':
    #from a large to many small subarray
    import sys
    X = np.load('normalized.npy')
    X = row_slice (X,int(sys.argv[1]))
    for i in range(int(sys.argv[1])):
        np.save('subnromalized{}.npy'.format(i),X[i])
