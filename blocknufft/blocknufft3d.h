#ifndef BLOCKNUFFT3D_H
#define BLOCKNUFFT3D_H

struct BlockNufft3DOptions {
	int N1,N2,N3; //uniform grid size
	int K1,K2,K3; //block size
	int M; //number of non-uniform points
	int num_threads; // The number of parallel threads to use
    double eps; //requested precision --> determines tau, oversamp factor and R
};

//x,y,z should be in range [0,2pi)
bool blocknufft3d(const BlockNufft3DOptions &opts,double *uniform_d,double *x,double *y,double *z,double *nonuniform_d);
void test_blocknufft3d(BlockNufft3DOptions &opts);

struct BlockSpread3DOptions {
	int N1o,N2o,N3o; //size of grid
	int K1,K2,K3; //block size
	int nspread1,nspread2,nspread3; //kernel size
	int M; //number of non-uniform points
	int num_threads; // The number of parallel threads to use
    double tau1,tau2,tau3; //spreading factor
};

bool blockspread3d(const BlockSpread3DOptions &opts,double *uniform_d,double *spread,double *x,double *y,double *z,double *nonunform_d);

/* MATLAB interface using MCWRAP
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
MCWRAP [ COMPLEX uniform_d[N1,N2,N3], COMPLEX spread[N1*2,N2*2,N3*2] ] = blocknufft3d( xyz[M,3], COMPLEX nonuniform_d[M,1], eps, N1, N2, N3, K1, K2, K3, num_threads )
    SET_INPUT M = size(xyz,1)
    SOURCES blocknufft3d.o qute.cpp
    MEXARGS -largeArrayDims -lm -lgomp -lfftw3 -lfftw3_threads
IMPORTANT: In order to get openmp to work properly you must generate blocknufft3d.o separately using g++
  > g++ -fopenmp -c blocknufft3d.cpp -fPIC -O3
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
*/
void blocknufft3d(int N1,int N2,int N3,int M,double *uniform_d,double *spread,double *xyz,double *nonuniform_d,double eps,int K1,int K2,int K3,int num_threads);


#endif // BLOCKNUFFT3D_H
