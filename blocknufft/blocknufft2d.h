#ifndef BLOCKNUFFT2D_H
#define BLOCKNUFFT2D_H

struct BlockNufft2DOptions {
	int N1,N2; //number of grid points
	int K1,K2; //block size
	int R1,R2; //kernel size
	int M; //number of non-uniform points
	int num_threads;
};

bool blockspread2d(const BlockNufft2DOptions &opts,double *out,double *x,double *y,double *d);
void test_blockspread2d(const BlockNufft2DOptions &opts);

#endif // BLOCKNUFFT2D_H

