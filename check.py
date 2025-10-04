import numpy as np
import time
x1 = np.load('eigs.npy')
x2 = np.load('eigvals.npy')
arr = np.load('arr1.npy')
t1 = time.time()
e1, e2 = np.linalg.eig(arr)
print('Elapsed time:', time.time()-t1,' seconds')
print('MAE:',np.average(np.sort(e1)-np.sort(x2)))
#print('Numpy eigvals', np.sort(e1))
#print('ScaLAPACK eigvals', np.sort(x2))
