# Testing ELPA


In this repository, I want to test ELPA to diagonalize a large matrix. This code reads a numpy array, 
using (`cnpy` library)[https://github.com/rogersce/cnpy]. Then diagonalizes it using ELPA and saves the eigenvalues and eigenvectors in `.npy` format. The comparison of the `numpy` diagonalized array and ELPA diagonalized array.

The name of the file is `src/read_numpy.cpp` (the naming should have been better but anyways :)). It was just for me to get familiarized with MPI programming, BLACS, and ELPA which is similar to ScaLAPACK. 


Note: I could not upload the Hamiltonian matrices as they were large. These matrices represented Hermitian matrices from CP2K calculations. 

To compile this code:

```bash
mkdir build
cd build
cmake ../
make
```

and to run this, you can try this using slurm:

```bash
#!/bin/sh -l
#SBATCH --account=alexeyak
##SBATCH --partition=valhalla  --qos=valhalla
##SBATCH --clusters=faculty
#SBATCH --partition=general-compute  --qos=general-compute
#SBATCH --clusters=ub-hpc
##SBATCH --partition=scavenger  --qos=scavenger
#SBATCH --time=00:10:00 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
##SBATCH --constraint=AVX2
#SBATCH --cpus-per-task=1
#SBATCH --mem=128000
#SBATCH --constraint=AVX512
###SBATCH --mail-user=mshakiba@buffalo.edu

module load foss scipy-bundle matplotlib h5py scikit-learn imkl
export LD_LIBRARY_PATH=/cvmfs/soft.ccr.buffalo.edu/versions/2023.01/easybuild/software/avx512/MPI/gcc/11.2.0/openmpi/4.1.1/scalapack/2.1.0-fb/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/cvmfs/soft.ccr.buffalo.edu/versions/2023.01/easybuild/software/avx512/Compiler/gcccore/11.2.0/python/3.9.6/lib:$LD_LIBRARY_PATH

date
mpirun -np 4 ./diagonalize
date
```

I'm running this on UB CCR. 

Numpy couldn't diagonalize the `7020x7020` matrix to compare with it. But here is a comparison of a smaller matrix of size `240x240`. 
It seems that ELPA is not good for smaller matrices. It is also noted for block diagonalization in some resources. Here is the comparison:
```python
import numpy as np
>>> x_numpy = np.load('build/numpy_eigs.npy')
>>> x_numpy
array([[-0.10269306, -0.06998295, -0.15125062, ...,  0.03415571,
        -0.03152887, -0.00101298],
       [ 0.00867632, -0.03313664,  0.00342549, ..., -0.05147768,
         0.03518987,  0.03836706],
       [ 0.03229798, -0.00122085,  0.0257353 , ..., -0.02638489,
         0.00378227,  0.05392513],
       ...,
       [ 0.03694584,  0.01087838, -0.08028733, ...,  0.03389977,
        -0.01183407, -0.01019853],
       [-0.03558114, -0.05707275,  0.09126327, ..., -0.00321802,
        -0.01338405,  0.10150728],
       [-0.01866054, -0.00583954,  0.0306635 , ..., -0.06781775,
        -0.00844388, -0.0609266 ]])
>>> x_elpa = np.load('build/eigs_elpa.npy')
>>> x_elpa
array([[-0.10269306,  0.00867632,  0.03229798, ...,  0.03694584,
        -0.03558114, -0.01866054],
       [-0.06998295, -0.03313664, -0.00122085, ...,  0.01087838,
        -0.05707275, -0.00583954],
       [ 0.15125062, -0.00342549, -0.0257353 , ...,  0.08028733,
        -0.09126327, -0.0306635 ],
       ...,
       [ 0.03969885,  0.22210315,  0.04108508, ..., -0.02585942,
        -0.03888838,  0.03563613],
       [-0.01708045, -0.13082427, -0.00371917, ...,  0.06215819,
         0.12645585, -0.12856991],
       [ 0.01939967,  0.14855288,  0.00699558, ...,  0.07327291,
         0.1407157 , -0.13958312]])
```
