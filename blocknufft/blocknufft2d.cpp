#include "blocknufft2d.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "qute.h"
#include "omp.h"

double eval_kernel(double x,double y) {
	return 1+0*x+0*y;
}

bool blockspread2d(const BlockNufft2DOptions &opts,double *out,double *x,double *y,double *d) {
	int M=opts.M;
	int K1=opts.K1,K2=opts.K2;
	int N1=opts.N1,N2=opts.N2; int N1N2=N1*N2;
	int R1=opts.R1,R2=opts.R2;
	if ((R1>K1)||(R2>K2)) {
		printf("R cannot be bigger than K\n");
		return false;
	}
	for (int m=0; m<M; m++) {
		if ((x[m]<0)||(x[m]>=opts.N1)) {
			printf("x is out of bounds: x[%d]=%g\n",m,x[m]);
			return false;
		}
		if ((y[m]<0)||(y[m]>=opts.N2)) {
			printf("y is out of bounds: y[%d]=%g\n",m,y[m]);
			return false;
		}
	}

	for (int n=0; n<N1N2; n++) out[n]=0;

	printf("setting up input blocks... ");
	QTime timer0; timer0.start();
	int num_blocks_x=ceil(opts.N1*1.0/K1);
	int num_blocks_y=ceil(opts.N2*1.0/K2);
	int num_blocks=num_blocks_x*num_blocks_y;
	int *input_block_counts=(int *)malloc(sizeof(int)*num_blocks); for (int i=0; i<num_blocks; i++) input_block_counts[i]=0;
	for (int m=0; m<M; m++) {
		int b1=(int)(x[m]/K1);
		int c1=x[m]-b1*K1;
		int b2=(int)(y[m]/K2);
		int c2=y[m]-b2*K2;
		bool left_side=false,right_side=false;
		bool top_side=false,bottom_side=false;
		if ((c1>=K1-R1/2)&&(b1+1<num_blocks_x)) right_side=true;
		if ((c1<=-R1/2+R1-1)&&(b1-1>=0)) left_side=true;
		if ((c2>=K2-R2/2)&&(b2+1<num_blocks_y)) top_side=true;
		if ((c2<=-R2/2+R2-1)&&(b2-1>=0)) bottom_side=true;

		input_block_counts[b1+num_blocks_x*b2]++;
		if (left_side) input_block_counts[(b1-1)+num_blocks_x*b2]++;
		if (right_side) input_block_counts[(b1+1)+num_blocks_x*b2]++;
		if (bottom_side) input_block_counts[b1+num_blocks_x*(b2-1)]++;
		if (top_side) input_block_counts[b1+num_blocks_x*(b2+1)]++;
		if ((left_side)&&(bottom_side)) input_block_counts[(b1-1)+num_blocks_x*(b2-1)]++;
		if ((left_side)&&(top_side)) input_block_counts[(b1-1)+num_blocks_x*(b2+1)]++;
		if ((right_side)&&(bottom_side)) input_block_counts[(b1+1)+num_blocks_x*(b2-1)]++;
		if ((right_side)&&(top_side)) input_block_counts[(b1+1)+num_blocks_x*(b2+1)]++;
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
	for (int n2=0; n2<N2; n2++) {
		for (int n1=0; n1<N1; n1++) {
			int b1=n1/K1;
			int b2=n2/K2;
			output_block_counts[b1+num_blocks_x*b2]++;
		}
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
	double *input_y=(double *)malloc(sizeof(double)*input_size);
	double *input_d=(double *)malloc(sizeof(double)*input_size);
	int *input_ii=(int *)malloc(sizeof(int)*num_blocks);
	for (int ii=0; ii<num_blocks; ii++) input_ii[ii]=0;
	for (int m=0; m<M; m++) {
		int b1=(int)(x[m]/K1);
		int c1=x[m]-b1*K1;
		int b2=(int)(y[m]/K2);
		int c2=y[m]-b2*K2;
		bool left_side=false,right_side=false;
		bool top_side=false,bottom_side=false;
		if ((c1>=K1-R1/2)&&(b1+1<num_blocks_x)) right_side=true;
		if ((c1<=-R1/2+R1-1)&&(b1-1>=0)) left_side=true;
		if ((c2>=K2-R2/2)&&(b2+1<num_blocks_y)) top_side=true;
		if ((c2<=-R2/2+R2-1)&&(b2-1>=0)) bottom_side=true;

		{
			int bb=b1+num_blocks_x*b2;
			int jj=input_block_indices[bb]+input_ii[bb];
			input_x[jj]=x[m];
			input_y[jj]=y[m];
			input_d[jj]=d[m];
			input_ii[bb]++;
		}
		if (left_side) {
			int bb=(b1-1)+num_blocks_x*b2;
			int jj=input_block_indices[bb]+input_ii[bb];
			input_x[jj]=x[m];
			input_y[jj]=y[m];
			input_d[jj]=d[m];
			input_ii[bb]++;
		}
		if (right_side) {
			int bb=(b1+1)+num_blocks_x*b2;
			int jj=input_block_indices[bb]+input_ii[bb];
			input_x[jj]=x[m];
			input_y[jj]=y[m];
			input_d[jj]=d[m];
			input_ii[bb]++;
		}
		if (bottom_side) {
			int bb=(b1)+num_blocks_x*(b2-1);
			int jj=input_block_indices[bb]+input_ii[bb];
			input_x[jj]=x[m];
			input_y[jj]=y[m];
			input_d[jj]=d[m];
			input_ii[bb]++;
		}
		if (top_side) {
			int bb=(b1)+num_blocks_x*(b2+1);
			int jj=input_block_indices[bb]+input_ii[bb];
			input_x[jj]=x[m];
			input_y[jj]=y[m];
			input_d[jj]=d[m];
			input_ii[bb]++;
		}
		if ((left_side)&&(bottom_side)) {
			int bb=(b1-1)+num_blocks_x*(b2-1);
			int jj=input_block_indices[bb]+input_ii[bb];
			input_x[jj]=x[m];
			input_y[jj]=y[m];
			input_d[jj]=d[m];
			input_ii[bb]++;
		}
		if ((left_side)&&(top_side)) {
			int bb=(b1-1)+num_blocks_x*(b2+1);
			int jj=input_block_indices[bb]+input_ii[bb];
			input_x[jj]=x[m];
			input_y[jj]=y[m];
			input_d[jj]=d[m];
			input_ii[bb]++;
		}
		if ((right_side)&&(bottom_side)) {
			int bb=(b1+1)+num_blocks_x*(b2-1);
			int jj=input_block_indices[bb]+input_ii[bb];
			input_x[jj]=x[m];
			input_y[jj]=y[m];
			input_d[jj]=d[m];
			input_ii[bb]++;
		}
		if ((right_side)&&(top_side)) {
			int bb=(b1+1)+num_blocks_x*(b2+1);
			int jj=input_block_indices[bb]+input_ii[bb];
			input_x[jj]=x[m];
			input_y[jj]=y[m];
			input_d[jj]=d[m];
			input_ii[bb]++;
		}

	}
	printf("Elapsed: %d ms\n",timerA.elapsed());

	printf("spreading... ");
	QTime timerC; timerC.start();
	omp_set_num_threads(opts.num_threads);
	#pragma omp parallel
	{
		#pragma omp for
		for (int cc=0; cc<num_blocks; cc++) {
			int cc1=cc%num_blocks_x;
			int cc2=cc/num_blocks_x;
			int jj=input_block_indices[cc];
			int tmp=jj+input_block_counts[cc];
			while (jj<tmp) {
				double x0=input_x[jj];
				double y0=input_y[jj];
				double d0=input_d[jj];
				int aa1=((int)x0)-R1/2;
				int bb1=aa1+R1;
				int aa2=((int)y0)-R2/2;
				int bb2=aa2+R2;
				for (int kk2=aa2; kk2<bb2; kk2++) {
					if ((cc2*K2<=kk2)&&(kk2<(cc2+1)*K2)&&(kk2<N2)) {
						for (int kk1=aa1; kk1<bb1; kk1++) {
							if ((cc1*K1<=kk1)&&(kk1<(cc1+1)*K1)&&(kk1<N1)) {
								out[kk1+N1*kk2]+=d0*eval_kernel(kk1-x0,kk2-y0);
							}
						}
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
	free(input_y);
	free(input_d);
	free(input_ii);
	printf("Elapsed: %d ms\n",timerD.elapsed());

	return true;
}

void test_blockspread2d(const BlockNufft2DOptions &opts)
{
	printf("test_blockspread2d: preparing.\n");
	double *x=(double *)malloc(sizeof(double)*opts.M);
	double *y=(double *)malloc(sizeof(double)*opts.M);
	double *d=(double *)malloc(sizeof(double)*opts.M);
	double *out=(double *)malloc(sizeof(double)*opts.N1*opts.N2);

	for (int m=0; m<opts.M; m++) {
		x[m]=qrand()%opts.N1;
		y[m]=qrand()%opts.N2;
		d[m]=1;
	}

	QTime timer; timer.start();
	blockspread2d(opts,out,x,y,d);
	printf("Time for blockspread2d: %d ms\n\n",timer.elapsed());

	free(x);
	free(y);
	free(d);
	free(out);
}
