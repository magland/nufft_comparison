#include "blocknufft1d.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "qute.h"
#include "omp.h"

double eval_kernel(double x) {
	return 1+0*x;
}

bool blockspread1d(const BlockNufft1DOptions &opts,double *out,double *x,double *d) {
	int M=opts.M;
	int K=opts.K;
	int N=opts.N;
	int R=opts.R;
	if (R>K) {
		printf("R cannot be bigger than K\n");
		return false;
	}
	for (int m=0; m<M; m++) {
		if ((x[m]<0)||(x[m]>=opts.N)) {
			printf("x is out of bounds: x[%d]=%g\n",m,x[m]);
			return false;
		}
	}

	for (int n=0; n<N; n++) out[n]=0;

	printf("setting up input blocks... ");
	QTime timer0; timer0.start();
	int num_blocks=ceil(opts.N*1.0/K);
	int *input_block_counts=(int *)malloc(sizeof(int)*num_blocks); for (int i=0; i<num_blocks; i++) input_block_counts[i]=0;
	for (int m=0; m<M; m++) {
		int b1=(int)(x[m]/K);
		int c1=x[m]-b1*K;
		input_block_counts[b1]++;
		if ((c1>=opts.K-R/2)&&(b1+1<num_blocks)) {
			input_block_counts[b1+1]++;
		}
		if ((c1<=-R/2+R-1)&&(b1-1>=0)) {
			input_block_counts[b1-1]++;
		}
	}
	int input_size=0;
	int *input_block_indices=(int *)malloc(sizeof(int)*num_blocks);
	for (int i=0; i<num_blocks; i++) {
		input_block_indices[i]=input_size;
		input_size+=input_block_counts[i];
	}
	printf("Elapsed: %d ms\n",timer0.elapsed());

	QTime timerA; timerA.start();
	printf("setting up output blocks... ");
	int *output_block_counts=(int *)malloc(sizeof(int)*num_blocks); for (int i=0; i<num_blocks; i++) output_block_counts[i]=0;
	for (int n=0; n<N; n++) {
		int b1=n/K;
		output_block_counts[b1]++;
	}
	int output_size=0;
	int *output_block_indices=(int *)malloc(sizeof(int)*num_blocks);
	for (int i=0; i<num_blocks; i++) {
		output_block_indices[i]=output_size;
		output_size+=output_block_counts[i];
	}
	printf("Elapsed: %d ms\n",timerA.elapsed());

	printf("setting input... ");
	QTime timerB; timerB.start();
	double *input_x=(double *)malloc(sizeof(double)*input_size);
	double *input_d=(double *)malloc(sizeof(double)*input_size);
	int *input_ii=(int *)malloc(sizeof(int)*num_blocks);
	for (int ii=0; ii<num_blocks; ii++) input_ii[ii]=0;
	for (int m=0; m<M; m++) {
		int b1=(int)(x[m]/K);
		int c1=x[m]-b1*K;
		int jj=input_block_indices[b1]+input_ii[b1];
		input_x[jj]=x[m];
		input_d[jj]=d[m];
		input_ii[b1]++;
		if ((c1>=opts.K-R/2)&&(b1+1<num_blocks)) {
			int jj=input_block_indices[b1+1]+input_ii[b1+1];
			input_x[jj]=x[m];
			input_d[jj]=d[m];
			input_ii[b1+1]++;
		}
		if ((c1<=-R/2+R-1)&&(b1-1>=0)) {
			int jj=input_block_indices[b1-1]+input_ii[b1-1];
			input_x[jj]=x[m];
			input_d[jj]=d[m];
			input_ii[b1-1]++;
		}
	}
	printf("Elapsed: %d ms\n",timerA.elapsed());

	printf("spreading... ");
	QTime timerC; timerC.start();
	omp_set_num_threads(opts.num_threads);
	#pragma omp parallel
	{
		#pragma omp for
		for (int bb=0; bb<num_blocks; bb++) {
			int jj=input_block_indices[bb];
			int tmp=jj+input_block_counts[bb];
			while (jj<tmp) {
				double x0=input_x[jj];
				double d0=input_d[jj];
				int aa1=((int)x0)-R/2;
				int aa2=aa1+R;
				for (int kk=aa1; kk<aa2; kk++) {
					if ((bb*K<=kk)&&(kk<(bb+1)*K)&&(kk<N)) {
						out[kk]+=d0*eval_kernel(kk-x0);
					}
				}
				jj++;
			}
		}
	}
	printf("Elapsed: %d ms\n",timerA.elapsed());

	printf("freeing... ");
	QTime timerD; timerD.start();
	free(input_block_counts);
	free(input_block_indices);
	free(output_block_counts);
	free(output_block_indices);
	free(input_x);
	free(input_d);
	free(input_ii);
	printf("Elapsed: %d ms\n",timerD.elapsed());

	return true;
}


void test_blockspread1d(const BlockNufft1DOptions &opts)
{
	printf("test_blockspread2d: preparing.\n");
	double *x=(double *)malloc(sizeof(double)*opts.M);
	double *d=(double *)malloc(sizeof(double)*opts.M);
	double *out=(double *)malloc(sizeof(double)*opts.N);

	for (int m=0; m<opts.M; m++) {
		x[m]=qrand()%opts.N;
		d[m]=1;
	}

	QTime timer; timer.start();
	blockspread1d(opts,out,x,d);
	printf("Time for blockspread1d: %d ms\n\n",timer.elapsed());

	free(x);
	free(d);
	free(out);
}
