#ifndef BLOCKNUFFT1D_H
#define BLOCKNUFFT1D_H

struct BlockNufft1DOptions {
	int N; //number of grid points
	int K; //block size
	int R; //kernel size
	int M; //number of non-uniform points
	int num_threads;
};

bool blockspread1d(const BlockNufft1DOptions &opts,double *out,double *x,double *d);
void test_blockspread1d(const BlockNufft1DOptions &opts);

#endif // BLOCKNUFFT1D_H

