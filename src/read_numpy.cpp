#include <iostream>
#include <stdio.h>
#include <vector>
#include <mpi.h>
#include <cnpy.h>
#include <mkl_types.h>
#include <mkl_scalapack.h>
#include <mkl_blacs.h>
#include <mkl_pblas.h>
#include <mkl_lapacke.h>
#include <cmath>
#include <elpa/elpa.h>

using namespace std;
using namespace cnpy;
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int nprocs, myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    double val;
    NpyArray arr;
    int arr_size;
    double data[2];
    cout << " Hello world! " << myrank << endl;
    // Read the numpy array
    if (myrank==0) {
    arr = npy_load("arr1.npy");
    //double* data;
    double* data1 = arr.data<double>();
    //double* data = arr.data<double>();
    arr_size = arr.shape[0]*arr.shape[1];
    cout << "Rank " << myrank <<  endl;
//    data[0] = 100.0;
//    data[1] = 10.0;
    //val = 0.918;
//    val = A[0];
    } 
    // broadcast the values and the array size
    //MPI_Bcast(&val, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&arr_size, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    //MPI_Bcast(data, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
//    cout << "myrank : " << myrank << "  val " << val << endl;
    cout << "myrank : " << myrank << "  arr_size " << arr_size << endl;
//    cout << "myrank : " << myrank << "  data[0] " << data[0] << endl;
//    cout << "myrank : " << myrank << "  data[1] " << data[1] << endl;
//    cout << "Now on rank " << myrank << endl;
    // create a data1 for each processor
    double data1[arr_size];
    // read the data in the manager rank
    //if (myrank==0) {
    arr = npy_load("arr1.npy");
    //double* data;
    cout << "Flag x npload" << endl;
    double* A = arr.data<double>();
    cout << "Flag y extracting data" << endl;
    //double* data = arr.data<double>();
    for (int i=0; i<arr_size; i++){
    data1[i] = A[i];
    }
    //}
    // broadcast it to other processors
    MPI_Bcast(data1, arr_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    cout << "myrank : " << myrank << "  data1 " << data1[myrank+100] << endl;


    MPI_Barrier(MPI_COMM_WORLD);
    // setting up blacs grid

    int n = static_cast<int>(std::sqrt(arr_size));
    int nb = 20;   // Block size
    int context, myrow, mycol, nprow, npcol, myrank1;
    int nproc1;
    int Zero=0, One=1, MOne=-1;
    blacs_pinfo(&myrank1, &nproc1);
    cout << "Process number: " << myrank1 << endl;
    //npcol = 1;

    blacs_get(&MOne, &Zero, &context);
    cout << "BLACS context: " << context << endl;
    nprow = static_cast<int>(std::sqrt(nprocs));
    npcol = static_cast<int>(std::sqrt(nprocs));
    //nprow = 2, npcol = 2;
    blacs_gridinit(&context, "Row", &nprow, &npcol);
    int context_global;
    int One1 = 1;
//    //blacs_gridinit(&context_global, "Row", &One1, &One1);
    blacs_gridinfo(&context, &nprow, &npcol, &myrow, &mycol);
//
    cout << "Process (" << myrow << ", " << mycol << ") out of (" << nprow << ", " << npcol << ") in the BLACS process grid. myrank1:" << myrank1 << " myrank:" << myrank << " context:" << context<< endl;
//    
    int desc[9];
    int info;
    int mpA = numroc_(&n, &nb, &myrow, &Zero, &nprow);
    int npA = numroc_(&n, &nb, &mycol, &Zero, &npcol);
    int lld = mpA;
    cout << "Flag 0 lld: " << lld << endl;
//    //int lld = (n + nprow - 1) / nprow;  // the local leading dimension
    descinit_(desc, &n, &n, &nb, &nb, &Zero, &Zero, &context, &lld, &info);
    cout << "Flag 1" << endl;
//    //int descz[9];
//    //descinit_(descz, &n, &n, &mb, &nb, &irsrc, &icsrc, &context, &lld, &info);
//    //descinit_(descz, &n, &n, &nb, &nb, &Zero, &Zero, &context, &lld, &info);
//
    if (info != 0) {
        std::cerr << "Error in descinit_: " << info << std::endl;
        MPI_Abort(MPI_COMM_WORLD, info);
    }
//
//    // Global eigenvalues and eigenvectors
//    
//    
    // Eigenvalue storage
    std::vector<double> w(n);
//    // Initialize eigenvector storage
    std::vector<double> z(n * n);  // Assuming z has the same size as the input matrix
    int desc_global[9];
    //if (myrank!=0){
    //double data1[arr_size];
    //}
       //descinit_(desc_global, &n, &n, &n, &n, &Zero, &Zero, &context_global, &lld, &info);
       descinit_(desc_global, &n, &n, &n, &n, &Zero, &Zero, &context, &n, &info);
    //}
    // Initialize local eigenvectors
    double data1_loc[lld*lld];
//    //std::vector<double> wloc(lld);
    std::vector<double> zloc(lld*lld);
    //double zloc[lld*lld];
//
//    // Workspace query
    MKL_INT info1;
    MKL_INT lwork = -1;
//    //MKL_INT liwork = -1;
    MKL_INT iz = 1, jz = 1;
    std::vector<double> work(1);
    std::vector<MKL_INT> iwork(1);
//
//
    // distribute matrix the global matrix to local matrices with pdgemr2d
    cout << "Flag 2" << endl;
    pdgemr2d_(&n, &n, data1, &One, &One, desc_global, data1_loc, &One, &One, desc, &context);
    cout << "Flag 3" << endl;
    cout << "Flag 4 values of data1_loc" << data1_loc[0] << endl;
//    //pdsyev_("V", "U", &n, data1_loc, &One, &One, desc, w.data(), zloc.data(), &iz, &jz, desc_global, work.data(), &lwork, &info1);
    pdsyev_("V", "U", &n, data1_loc, &One, &One, desc, w.data(), zloc.data(), &iz, &jz, desc, work.data(), &lwork, &info1);
//    //pdsyev_("V", "U", &n, data1, &One, &One, desc, w.data(), work.data(), &lwork, iwork.data(), &liwork, &info1);
//    //pdsyev_(   ,    ,  n,     a,     , * ja, desca,       w,       ble*z,     iz, const MKL_INT* jz, const MKL_INT* descz, double* work, const MKL_INT* lwork, MKL_INT* info);
//
    lwork = static_cast<MKL_INT>(work[0]);
//    //liwork = iwork[0];
//
    work.resize(lwork);
//    //iwork.resize(liwork);
//
//    //pdsyev_("V", "U", &n, data1, &One, &One, desc, w.data(), work.data(), &lwork, iwork.data(), &liwork, &info1);
    pdsyev_("V", "U", &n, data1_loc, &One, &One, desc, w.data(), zloc.data(), &iz, &jz, desc, work.data(), &lwork, &info1);
//    //pdsyev_("V", "U", &n, data1, &One, &One, desc, w.data(), z.data(), &iz, &jz, descz, work.data(), &lwork, &info1);
//
    pdgemr2d_(&n, &n, zloc.data(), &One, &One, desc, z.data(), &One, &One, desc_global, &context);

    if (info1 == 0) {
        //NpyArray 
        //std::stringstream ss;
        //ss << "eigs_" << myrank << ".npy";
        //std::string filename = ss.str();
        //cnpy::npy_save(ss.str(),&z[0],{n,n},"w");
        //cnpy::npy_save("eigs.npy",&z[0],{n,n},"w");
        if (myrank == 0) {
            cnpy::npy_save("eigs.npy",&z[0],{n,n},"w");
            cnpy::npy_save("eigvals.npy",&w[0],{n},"w");
            //std::cout << "Eigenvalues:\n";
            //for (int i = 0; i < n; ++i) {
            //    std::cout << w[i] << "\n";
            //}
        }
    } else {
        std::cerr << "Diagonalization failed with error " << info1 << "\n";
    }

//    pdgemr2d_(&n, &n, data1, &One, &One, desc_global, data1_loc, &One, &One, desc, &context);
//    int error;
//    elpa_t handle;
//    elpa_init(20211125);
//    handle = elpa_allocate(&error);
//    if (error !=ELPA_OK){
//       cout << "Could not initialize ELPA!!!!" << endl;
//       MPI_Finalize();
//       return 1;
//    }
//
//    elpa_set(handle, "na", n, &error);
//    elpa_set(handle, "nev", n, &error);
//    elpa_set(handle, "nprow", nprow, &error);
//    elpa_set(handle, "npcol", npcol, &error);
//    elpa_set(handle, "nblk", nb, &error);
//    elpa_set(handle, "mpi_comm_parent", MPI_Comm_c2f(MPI_COMM_WORLD), &error);
//    elpa_set(handle, "process_row", myrow, &error);
//    elpa_set(handle, "local_nrows", mpA, &error);
//    elpa_set(handle, "local_ncols", npA, &error);
//    elpa_set(handle, "process_col", mycol, &error);
//
//    error = elpa_setup(handle);
//
//    double W[n];
//    double Z[mpA*mpA];
//    double Z_[n*n];
//
//    elpa_set(handle, "solver", ELPA_SOLVER_2STAGE, &error);
//    elpa_eigenvectors(handle, data1_loc, W, Z, &error);
//
//    pdgemr2d_(&n, &n, Z, &One, &One, desc, Z_, &One, &One, desc_global, &context);
//    MPI_Barrier(MPI_COMM_WORLD);
//
//        if (myrank == 0) {
//            cnpy::npy_save("eigs_elpa.npy",&Z_[0],{n,n},"w");
//            cnpy::npy_save("eigvals_elpa.npy",&W[0],{n},"w");
//            //std::cout << "ELPA Eigenvalues: -------------------------------------------------- \n";
//            
//            //for (int i = 0; i < n; ++i) {
//            //    std::cout << w[i] << "\n";
//            //}
//        }
//
//    elpa_deallocate(handle, &error);
//    elpa_uninit(&error);

    blacs_gridexit(&context);

//    cout << "The size of the matrix n: " << n << endl;
//    cout << "The block size is " << nb << "x" << nb << endl;
//    cout << "The local leading dimension is : " << lld << endl;


    MPI_Finalize();
    return 0;
}

