import numpy as np
import time
x1 = np.load('eigs.npy')
x2 = np.load('eigvals.npy')
arr = np.load('arr1.npy')

xelp1 = np.load('eigs_elpa.npy')
xelp2 = np.load('eigvals_elpa.npy')

#t1 = time.time()
#e1, e2 = np.linalg.eig(arr)
#np.save('numpy_eigvals.npy', e1)
#np.save('numpy_eigs.npy', e2)
e1 = np.load('numpy_eigvals.npy')
e2 = np.load('numpy_eigs.npy')
#print('Elapsed time:', time.time()-t1,' seconds')
print('ScaLAPACK MAE:',np.average(np.abs(np.sort(e1)-np.sort(x2))))
print('ELPA ScaLAPACK MAE:',np.average(np.abs(np.sort(xelp2)-np.sort(x2))))
print('ELPA MAE:',np.average(np.abs(np.sort(e1)-np.sort(xelp2))))

print('ELPA eigvals', np.sort(xelp2))
print('Numpy eigvals', np.sort(e1))
print('ScaLAPACK eigvals', np.sort(x2))



