#include "blocknufft3d.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "qute.h"
#include "omp.h"
#include "fftw3.h"
#include "besseli.h"
#include "unistd.h"

//#define NUFFT_ANALYTIC_CORRECTION

// The following is needed because incredibly C++ does not seem to have a round() function
#define ROUND_2_INT(f) ((int)(f >= 0.0 ? (f + 0.5) : (f - 0.5)))

// Info for the spreading kernel
struct KernelInfo {
	int kernel_type;

	int nspread;

	//for gaussian
	double tau;
	double *lookup_exp;

	//for KB
	double W;
	double beta;
};

// This is the data needed for the spreading
struct BlockSpread3DData {
	int M; // Number of non-uniform points
	int N1o,N2o,N3o; // Oversampled dimensions
	double *x,*y,*z; // non-uniform locations
	double *nonuniform_d; // non-uniform data
	double *uniform_d; // uniform data
};

//Redundant?
struct BlockSpread3DOptions {
	int N1o,N2o,N3o; //size of grid
	int M; //number of non-uniform points
};

// These are the implementation routines for spreading
bool check_valid_inputs(BlockSpread3DData &D);
void define_block_ids_and_location_codes(BlockSpread3DData &D);
void compute_nonuniform_block_counts(BlockSpread3DData &D);
void compute_sizes_and_block_indices(BlockSpread3DData &D);
void set_working_nonuniform_data(BlockSpread3DData &D);
void do_spreading(BlockSpread3DData &D,const KernelInfo &KK1,const KernelInfo &KK2,const KernelInfo &KK3);
void set_uniform_data(BlockSpread3DData &D);

// Here's the spreading!
bool blockspread3d(const BlockSpread3DOptions &opts,const KernelInfo &KK1,const KernelInfo &KK2,const KernelInfo &KK3,double *uniform_d,double *x,double *y,double *z,double *nonuniform_d) {
	QTime timer0;
	printf("Starting blockspread3d...\n");

	// Transfer opts and other parameters to D
	BlockSpread3DData D;
	D.N1o=opts.N1o; D.N2o=opts.N2o; D.N3o=opts.N3o;
	D.M=opts.M;
	D.x=x; D.y=y; D.z=z;
	D.nonuniform_d=nonuniform_d;
	D.uniform_d=uniform_d;

	//Check to see if we have valid inputs
	printf("Checking inputs...\n"); timer0.start();
	if (!check_valid_inputs(D)) {
		for (int i=0; i<D.N1o*D.N2o*D.N3o*2; i++) D.uniform_d[i]=0;
		return false;
	}
	printf("  --- Elapsed: %d ms\n",timer0.elapsed());

	printf("spreading...\n"); timer0.start();
	do_spreading(D,KK1,KK2,KK3);
	printf("  --- Elapsed: %d ms\n",timer0.elapsed());

	return true;
}

// A couple implementation routines for nufft
bool do_fft_3d(int N1,int N2,int N3,double *out,double *in,int num_threads=1);
void do_fix_3d(int N1,int N2,int N3,int M,KernelInfo &KK1,KernelInfo &KK2,KernelInfo &KK3,double oversamp,double *out,double *out_oversamp_hat);

struct BlockData {
	int xmin,ymin,zmin;
	int xmax,ymax,zmax;
	int M;
	int N1o,N2o,N3o;
	double *x,*y,*z;
	double *nonuniform_d;
	double *uniform_d;
	int jj;
};

//set a lookup table for fast Gaussian spreading
#define MAX_LOOKUP_EXP 200
double s_lookup_exp1[MAX_LOOKUP_EXP];
double s_lookup_exp2[MAX_LOOKUP_EXP];
double s_lookup_exp3[MAX_LOOKUP_EXP];
void setup_lookup_exp1(double tau) {
	printf("SETUP LOOKUP EXP1 tau=%g\n",tau);
	for (int i=0; i<MAX_LOOKUP_EXP; i++) {
		s_lookup_exp1[i]=exp(-i*i*tau);
	}
}
void setup_lookup_exp2(double tau) {
	for (int i=0; i<MAX_LOOKUP_EXP; i++) {
		s_lookup_exp2[i]=exp(-i*i*tau);
	}
}
void setup_lookup_exp3(double tau) {
	for (int i=0; i<MAX_LOOKUP_EXP; i++) {
		s_lookup_exp3[i]=exp(-i*i*tau);
	}
}

// Here's the nufft!
bool blocknufft3d(const BlockNufft3DOptions &opts,double *out,double *x,double *y,double *z,double *d) {

	if (fork()==0) {
		printf("\n\n\nI've been forked!!!! Now I will die!!!!!\n\n\n");
		exit(0);
	}

	QTime timer0;
	QTime timer_total; timer_total.start();

	printf("\nStarting blocknufft3d.\n");

	omp_set_num_threads(opts.num_threads);

	double oversamp;

	//set up the spreading kernels
	KernelInfo KK1,KK2,KK3;

	if (opts.kernel_type==KERNEL_TYPE_KB) {
		KK1.kernel_type=KERNEL_TYPE_KB;
		KK2.kernel_type=KERNEL_TYPE_KB;
		KK3.kernel_type=KERNEL_TYPE_KB;

		oversamp=2;
		//int nspread=10; double fac1=0.90,fac2=1.47;
		int nspread=12; double fac1=1,fac2=1;
		if (opts.eps>=1e-2) {
			nspread=4; fac1=0.75; fac2=1.71;
		}
		else if (opts.eps>=1e-4) {
			nspread=6; fac1=0.83; fac2=1.56;
		}
		else if (opts.eps>=1e-6) {
			nspread=8; fac1=0.89; fac2=1.45;
		}
		else if (opts.eps>=1e-8) {
			nspread=10; fac1=0.90; fac2=1.47;
		}
		else if (opts.eps>=1e-10) {
			nspread=12; fac1=0.92; fac2=1.51;
		}
		else if (opts.eps>=1e-12) {
			nspread=14; fac1=0.94; fac2=1.48;
		}
		else {
			nspread=16; fac1=0.94; fac2=1.46;
		}

		{
			KK1.nspread=nspread;
			KK1.W=KK1.nspread*fac1;
			double tmp0=KK1.W*KK1.W/4-0.8;
			if (tmp0<0) tmp0=0; //fix this?
			KK1.beta=M_PI*sqrt(tmp0)*fac2;
		}

		{
			KK2.nspread=nspread;
			KK2.W=KK2.nspread*fac1;
			double tmp0=KK2.W*KK2.W/4-0.8;
			if (tmp0<0) tmp0=0; //fix this?
			KK2.beta=M_PI*sqrt(tmp0)*fac2;
		}

		{
			KK3.nspread=nspread;
			KK3.W=KK3.nspread*fac1;
			double tmp0=KK3.W*KK3.W/4-0.8;
			if (tmp0<0) tmp0=0; //fix this?
			KK3.beta=M_PI*sqrt(tmp0)*fac2;
		}
	}
	else if (opts.kernel_type==KERNEL_TYPE_GAUSSIAN) {
		double eps=opts.eps * 10; //note: jfm multiplied by 100 here!
		oversamp=2; if (eps<= 1e-11) oversamp=3;
		int nspread=(int)(-log(eps)/(M_PI*(oversamp-1)/(oversamp-.5)) + .5) + 1; //the plus one was added -- different from docs -- aha!
		nspread=nspread*2; //we need to multiply by 2, because I consider nspread as the diameter
		double lambda=oversamp*oversamp * nspread/2 / (oversamp*(oversamp-.5));
		double tau=M_PI/lambda;
		printf("Using oversamp=%g, nspread=%d, tau=%g\n",oversamp,nspread,tau);

		KK1.kernel_type=KERNEL_TYPE_GAUSSIAN;
		KK2.kernel_type=KERNEL_TYPE_GAUSSIAN;
		KK3.kernel_type=KERNEL_TYPE_GAUSSIAN;


		KK1.tau=tau;
		KK1.nspread=nspread;
		setup_lookup_exp1(KK1.tau);
		KK1.lookup_exp=s_lookup_exp1;

		KK2.tau=tau;
		KK2.nspread=nspread;
		setup_lookup_exp2(KK2.tau);
		KK2.lookup_exp=s_lookup_exp2;

		KK3.tau=tau;
		KK3.nspread=nspread;
		setup_lookup_exp3(KK3.tau);
		KK3.lookup_exp=s_lookup_exp3;
	}
	else {
		printf("Unknown kernel type: %d\n",opts.kernel_type);
		return false;
	}

	int N1o=(int)(opts.N1*oversamp); int N2o=(int)(opts.N2*oversamp); int N3o=(int)(opts.N3*oversamp);

	printf("Allocating...\n"); timer0.start();
	double *out_oversamp=(double *)malloc(sizeof(double)*N1o*N2o*N3o*2);
	double *out_oversamp_hat=(double *)malloc(sizeof(double)*N1o*N2o*N3o*2);
	printf("  --- Elapsed: %d ms\n",timer0.elapsed());

	printf("Scaling coordinates...\n"); timer0.start();
	double factor_x=N1o/(2*M_PI);
	double factor_y=N2o/(2*M_PI);
	double factor_z=N3o/(2*M_PI);
	for (int ii=0; ii<opts.M; ii++) {
		x[ii]*=factor_x;
		y[ii]*=factor_y;
		z[ii]*=factor_z;
	}
	printf("  --- Elapsed: %d ms\n",timer0.elapsed());

	//create blocks
	printf("Creating blocks...\n"); timer0.start();
	int num_blocks_x=ceil(N1o*1.0/opts.K1);
	int num_blocks_y=ceil(N2o*1.0/opts.K2);
	int num_blocks_z=ceil(N3o*1.0/opts.K3);
	int num_blocks=num_blocks_x*num_blocks_y*num_blocks_z;
	BlockData block_data[num_blocks];
	{
		int bb=0;
		for (int i3=0; i3<num_blocks_z; i3++) {
			for (int i2=0; i2<num_blocks_y; i2++) {
				for (int i1=0; i1<num_blocks_x; i1++) {
					BlockData BD;
					BD.xmin=i1*opts.K1; BD.xmax=fmin((i1+1)*opts.K1-1,N1o-1);
					BD.ymin=i2*opts.K2; BD.ymax=fmin((i2+1)*opts.K2-1,N2o-1);
					BD.zmin=i3*opts.K3; BD.zmax=fmin((i3+1)*opts.K3-1,N3o-1);
					BD.N1o=BD.xmax-BD.xmin+1+KK1.nspread;
					BD.N2o=BD.ymax-BD.ymin+1+KK2.nspread;
					BD.N3o=BD.zmax-BD.zmin+1+KK3.nspread;
					BD.M=0;
					block_data[bb]=BD;
					bb++;
				}
			}
		}
	}
	printf("  --- Elapsed: %d ms\n",timer0.elapsed());

	//compute block counts
	printf("Computing block counts...\n"); timer0.start();
	for (int ii=0; ii<opts.M; ii++) {
		int i1=((int)(x[ii]))/opts.K1;
		int i2=((int)(y[ii]))/opts.K2;
		int i3=((int)(z[ii]))/opts.K3;
		block_data[i1+num_blocks_x*i2+num_blocks_x*num_blocks_y*i3].M++;
	}
	printf("  --- Elapsed: %d ms\n",timer0.elapsed());

	//allocate block data
	printf("Allocating block data...\n"); timer0.start();
	for (int bb=0; bb<num_blocks; bb++) {
		BlockData *BD=&(block_data[bb]);
		BD->x=(double *)malloc(sizeof(double)*BD->M);
		BD->y=(double *)malloc(sizeof(double)*BD->M);
		BD->z=(double *)malloc(sizeof(double)*BD->M);
		BD->nonuniform_d=(double *)malloc(sizeof(double)*BD->M*2);
		BD->uniform_d=(double *)malloc(sizeof(double)*BD->N1o*BD->N2o*BD->N3o*2);
		BD->jj=0;
	}
	printf("  --- Elapsed: %d ms\n",timer0.elapsed());

	//set block input data
	printf("Setting block input data...\n"); timer0.start();
	for (int ii=0; ii<opts.M; ii++) {
		int i1=((int)(x[ii]))/opts.K1;
		int i2=((int)(y[ii]))/opts.K2;
		int i3=((int)(z[ii]))/opts.K3;
		BlockData *BD=&(block_data[i1+num_blocks_x*i2+num_blocks_x*num_blocks_y*i3]);
		int jj=BD->jj;
		BD->x[jj]=x[ii]-BD->xmin+KK1.nspread/2;
		BD->y[jj]=y[ii]-BD->ymin+KK2.nspread/2;
		BD->z[jj]=z[ii]-BD->zmin+KK3.nspread/2;
		BD->nonuniform_d[jj*2]=d[ii*2];
		BD->nonuniform_d[jj*2+1]=d[ii*2+1];
		BD->jj++;
	}
	printf("  --- Elapsed: %d ms\n",timer0.elapsed());

	printf("Spreading...\n"); timer0.start();
	#pragma omp parallel
	{
		QTime timerBB; timerBB.start(); int num_blocks_in_this_thread=0;
		if (omp_get_thread_num()==0) printf("#################### Using %d threads (%d prescribed)\n",omp_get_num_threads(),opts.num_threads);
		#pragma omp for
		for (int bb=0; bb<num_blocks; bb++) {
			BlockData *BD=&(block_data[bb]);
			BlockSpread3DOptions sopts;
			sopts.N1o=BD->N1o; sopts.N2o=BD->N2o; sopts.N3o=BD->N3o;
			sopts.M=BD->M;
			QTime timerAA; timerAA.start();
			blockspread3d(sopts,KK1,KK2,KK3,BD->uniform_d,BD->x,BD->y,BD->z,BD->nonuniform_d);
			num_blocks_in_this_thread++;
			printf("TIME for single block:::: %d ms, time in this thread: %d ms, #blocks in thread: %d\n",timerAA.elapsed(),timerBB.elapsed(),num_blocks_in_this_thread);
		}
		if (omp_get_thread_num()==0) printf("#################### Used %d threads (%d prescribed)\n",omp_get_num_threads(),opts.num_threads);
	}
	double blockspread3d_time=timer0.elapsed();
	printf("  For blockspread3d: %d ms\n",timer0.elapsed());

	printf("Combining uniform data...\n"); timer0.start();
	int N1oN2oN3o=N1o*N2o*N3o;
	for (int ii=0; ii<N1oN2oN3o; ii++) {
		out_oversamp[ii*2]=0;
		out_oversamp[ii*2+1]=0;
	}
	for (int bb=0; bb<num_blocks; bb++) {
		BlockData *BD=&(block_data[bb]);
		int jjj=0;
		for (int j3=0; j3<BD->N3o; j3++) {
			int k3=(j3+BD->zmin-KK3.nspread/2+N3o)%N3o;
			int kkk3=k3*N1o*N2o;
			for (int j2=0; j2<BD->N2o; j2++) {
				int k2=(j2+BD->ymin-KK2.nspread/2+N2o)%N2o;
				int kkk2=kkk3+k2*N1o;
				for (int j1=0; j1<BD->N1o; j1++) {
					int k1=(j1+BD->xmin-KK1.nspread/2+N1o)%N1o;
					int kkk1=kkk2+k1;
					out_oversamp[kkk1*2]+=BD->uniform_d[jjj*2];
					out_oversamp[kkk1*2+1]+=BD->uniform_d[jjj*2+1];
					jjj++;
				}
			}
		}
	}
	printf("  --- Elapsed: %d ms\n",timer0.elapsed());

	//free block data
	printf("Freeing block data...\n"); timer0.start();
	for (int bb=0; bb<num_blocks; bb++) {
		BlockData *BD=&(block_data[bb]);
		free(BD->x);
		free(BD->y);
		free(BD->z);
		free(BD->nonuniform_d);
		free(BD->uniform_d);
	}
	printf("  --- Elapsed: %d ms\n",timer0.elapsed());

	double spreading_time=timer_total.elapsed();
	printf("  --- Total time for spreading: %d ms\n",(int)spreading_time);

	printf("fft...\n"); timer0.start();
	if (!do_fft_3d(N1o,N2o,N3o,out_oversamp_hat,out_oversamp,opts.num_threads)) {
		printf("problem in do_fft_3d\n");
		for (int ii=0; ii<opts.M; ii++) {
			x[ii]/=factor_x;
			y[ii]/=factor_y;
			z[ii]/=factor_z;
		}
		free(out_oversamp);
		free(out_oversamp_hat);
		return false;
	}
	double fft_time=timer0.elapsed();
	printf("  --- Elapsed: %d ms\n",timer0.elapsed());

	printf("fix...\n"); timer0.start();
	do_fix_3d(opts.N1,opts.N2,opts.N3,opts.M,KK1,KK2,KK3,oversamp,out,out_oversamp_hat);
	printf("  --- Elapsed: %d ms\n",timer0.elapsed());

	printf("Restoring coordinates...\n"); timer0.start();
	for (int ii=0; ii<opts.M; ii++) {
		x[ii]/=factor_x;
		y[ii]/=factor_y;
		z[ii]/=factor_z;
	}
	printf("  --- Elapsed: %d ms\n",timer0.elapsed());

	printf("free...\n"); timer0.start();
	free(out_oversamp_hat);
	free(out_oversamp);
	printf("  --- Elapsed: %d ms\n",timer0.elapsed());

	double total_time=timer_total.elapsed();
	double other_time=total_time-spreading_time-fft_time;

	printf("Elapsed time: %.3f seconds\n",total_time/1000);
	printf("   %.3f blockspread3d, %.3f other spreading, %.3f fft, %.3f other\n",blockspread3d_time/1000,(spreading_time-blockspread3d_time)/1000,fft_time/1000,other_time/1000);
	printf("   %.1f%% blockspread3d, %.1f%% other spreading, %.1f%% fft, %.1f%% other\n",blockspread3d_time/total_time*100,(spreading_time-blockspread3d_time)/total_time*100,fft_time/total_time*100,other_time/total_time*100);

	printf("done with blocknufft3d.\n");

	return true;
}

// This is the mcwrap interface
void blocknufft3d(int N1,int N2,int N3,int M,double *uniform_d,double *xyz,double *nonuniform_d,double eps,int K1,int K2,int K3,int num_threads,int kernel_type) {
	BlockNufft3DOptions opts;
	opts.eps=eps;
	opts.K1=K1; opts.K2=K2; opts.K3=K3;
	opts.N1=N1; opts.N2=N2; opts.N3=N3;
	opts.M=M;
	opts.num_threads=num_threads;
	opts.kernel_type=kernel_type;

	double *x=&xyz[0];
	double *y=&xyz[M];
	double *z=&xyz[2*M];
	blocknufft3d(opts,uniform_d,x,y,z,nonuniform_d);
}

/////////////////////////////////////////////
/////////////////////////////////////////////
/////////////////////////////////////////////
/////////////////////////////////////////////
/////////////////////////////////////////////
// Below here are the implementation guts  //
/////////////////////////////////////////////
/////////////////////////////////////////////
/////////////////////////////////////////////
/////////////////////////////////////////////

bool do_fft_3d(int N1,int N2,int N3,double *out,double *in,int num_threads) {
	if (num_threads>1) {
		fftw_init_threads();
		fftw_plan_with_nthreads(num_threads);
	}

	int N1N2N3=N1*N2*N3;
	fftw_complex *in2=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*N1N2N3);
	fftw_complex *out2=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*N1N2N3);
	for (int ii=0; ii<N1N2N3; ii++) {
		in2[ii][0]=in[ii*2];
		in2[ii][1]=in[ii*2+1];
	}
	fftw_plan p=fftw_plan_dft_3d(N3,N2,N1,in2,out2,FFTW_BACKWARD,FFTW_ESTIMATE); //do we need to do N3,N2,N1 backwards? Really? Wasted half of a day. Oh well, in the mean time I made sure the rest of my code was bug free.
	fftw_execute(p);
	for (int ii=0; ii<N1N2N3; ii++) {
		out[ii*2]=out2[ii][0];
		out[ii*2+1]=out2[ii][1];
	}
	fftw_free(in2);
	fftw_free(out2);

	if (num_threads>1) {
		fftw_cleanup_threads();
	}

	return true;
}

void evaluate_kernel_1d(double *out,double diff,int imin,int imax,const KernelInfo &KK) {

	if (KK.kernel_type==KERNEL_TYPE_GAUSSIAN) {
		//exp(-(dx-i)^2*tau) = exp(-dx^2*tau)*exp(2*dx*tau)^i*exp(-i*i*tau)
		double term1=exp(-diff*diff*KK.tau);
		double factor2=exp(2*diff*KK.tau);
		double factor2_inv=1/factor2;

		double term2=term1;
		for (int i=0; i<=imax; i++) {
			out[i-imin]=term2*KK.lookup_exp[i];
			term2*=factor2;
		}

		term2=term1*factor2_inv;
		for (int i=-1; i>=imin; i--) {
			out[i-imin]=term2*KK.lookup_exp[-i];
			term2*=factor2_inv;
		}
	}
	else if (KK.kernel_type==KERNEL_TYPE_KB) {
		for (int ii=imin; ii<=imax; ii++) {
			//out[ii-imin]=1;
			double x=diff-ii;
			double tmp1=1-(2*x/KK.W)*(2*x/KK.W);
			if (tmp1<0) {
				out[ii-imin]=0;
			}
			else {
				double y=KK.beta*sqrt(tmp1);
				out[ii-imin]=besseli0_approx(y);
			}
		}
	}
}

void do_fix_3d(int N1,int N2,int N3,int M,KernelInfo &KK1,KernelInfo &KK2,KernelInfo &KK3,double oversamp,double *out,double *out_oversamp_hat) {
	int N1b=(int)(N1*oversamp);
	int N2b=(int)(N2*oversamp);
	int N3b=(int)(N3*oversamp);

#ifdef NUFFT_ANALYTIC_CORRECTION

	double lambda=M_PI/KK.tau;
	printf("t1 = %.16f\n",M_PI * lambda / (N1b*N1b));

	double *correction_vals1=(double *)malloc(sizeof(double)*(N1/2+2));
	double *correction_vals2=(double *)malloc(sizeof(double)*(N2/2+2));
	double *correction_vals3=(double *)malloc(sizeof(double)*(N3/2+2));

	double t1=M_PI * lambda / (N1b*N1b);
	double t2=M_PI * lambda / (N2b*N2b);
	double t3=M_PI * lambda / (N3b*N3b);
	for (int i=0; i<N1/2+2; i++) {
		correction_vals1[i]=exp(-i*i*t1)*(lambda*sqrt(lambda))*M;
	}
	for (int i=0; i<N2/2+2; i++) {
		correction_vals2[i]=exp(-i*i*t2);
	}
	for (int i=0; i<N3/2+2; i++) {
		correction_vals3[i]=exp(-i*i*t3);
	}

	for (int i3=0; i3<N3; i3++) {
		int aa3=((i3+N3/2)%N3)*N1*N2; //this includes a shift of the zero frequency
		int bb3=0;
		double correction3=1/(lambda*sqrt(lambda));
		if (i3<(N3+1)/2) { //had to be careful here!
			bb3=i3*N1*oversamp*N2*oversamp;
			correction3=correction_vals3[i3];
		}
		else {
			bb3=(N3*oversamp-(N3-i3))*N1*oversamp*N2*oversamp;
			correction3=correction_vals3[N3-i3];
		}
		for (int i2=0; i2<N2; i2++) {
			int aa2=((i2+N2/2)%N2)*N1;
			int bb2=0;
			double correction2=1;
			if (i2<(N2+1)/2) {
				bb2=i2*N1*oversamp;
				correction2=correction_vals2[i2];
			}
			else {
				bb2=(N2*oversamp-(N2-i2))*N1*oversamp;
				correction2=correction_vals2[N2-i2];
			}
			aa2+=aa3;
			bb2+=bb3;
			correction2*=correction3;
			for (int i1=0; i1<N1; i1++) {
				int aa1=(i1+N1/2)%N1;
				int bb1=0;
				double correction1=1;
				if (i1<(N1+1)/2) {
					bb1=i1;
					correction1=correction_vals1[i1];
				}
				else {
					bb1=N1*oversamp-(N1-i1);
					correction1=correction_vals1[N1-i1];
				}
				aa1+=aa2;
				bb1+=bb2;
				correction1*=correction2;
				out[aa1*2]=out_oversamp_hat[bb1*2]/correction1;
				out[aa1*2+1]=out_oversamp_hat[bb1*2+1]/correction1;
			}
		}
	}
	free(correction_vals1);
	free(correction_vals2);
	free(correction_vals3);
#else

	double *xcorrection=(double *)malloc(sizeof(double)*(N1b*2));
	double *ycorrection=(double *)malloc(sizeof(double)*(N2b*2));
	double *zcorrection=(double *)malloc(sizeof(double)*(N3b*2));

	int correction_kernel_oversamp=2; //do not change this -- hard-coded below

	for (int ik=1; ik<=3; ik++) {
		KernelInfo *KK;
		double *correction;
		int Nb;
		if (ik==1) {KK=&KK1; Nb=N1b; correction=xcorrection;}
		if (ik==2) {KK=&KK2; Nb=N2b; correction=ycorrection;}
		if (ik==3) {KK=&KK3; Nb=N3b; correction=zcorrection;}
		int nspread_correction=KK->nspread*2;

		double kernel_A[nspread_correction]; double kernel_B[nspread_correction];
		evaluate_kernel_1d(kernel_A,0,-nspread_correction/2,-nspread_correction/2+nspread_correction-1,*KK);
		evaluate_kernel_1d(kernel_B,-0.5,-nspread_correction/2,-nspread_correction/2+nspread_correction-1,*KK);
		for (int ii=0; ii<Nb*2; ii++) correction[ii]=0;
		for (int dx=-nspread_correction/2; dx<-nspread_correction/2+nspread_correction; dx++) {
			{
				double phase_increment=dx*2*M_PI/Nb;
				double kernel_val=kernel_A[dx+nspread_correction/2]/correction_kernel_oversamp;
				double phase_factor=0;
				for (int xx=0; xx<=Nb/2; xx++) {
					correction[xx*2]+=kernel_val*cos(phase_factor);
					correction[xx*2+1]+=kernel_val*sin(phase_factor);
					phase_factor+=phase_increment;
				}
				phase_factor=-phase_increment;
				for (int xx=Nb-1; xx>Nb/2; xx--) {
					correction[xx*2]+=kernel_val*cos(phase_factor);
					correction[xx*2+1]+=kernel_val*sin(phase_factor);
					phase_factor-=phase_increment;
				}
			}
			{
				double phase_increment=(dx+0.5)*2*M_PI/Nb;
				double kernel_val=kernel_B[dx+nspread_correction/2]/correction_kernel_oversamp;
				double phase_factor=0;
				for (int xx=0; xx<=Nb/2; xx++) {
					correction[xx*2]+=kernel_val*cos(phase_factor);
					correction[xx*2+1]+=kernel_val*sin(phase_factor);
					phase_factor+=phase_increment;
				}
				phase_factor=-phase_increment;
				for (int xx=Nb-1; xx>Nb/2; xx--) {
					correction[xx*2]+=kernel_val*cos(phase_factor);
					correction[xx*2+1]+=kernel_val*sin(phase_factor);
					phase_factor-=phase_increment;
				}
			}
		}
	}

	for (int i3=0; i3<N3; i3++) {
		int aa3=((i3+N3/2)%N3); //this includes a shift of the zero frequency
		int bb3=0;
		if (i3<(N3+1)/2) { //had to be careful here!
			bb3=i3;
		}
		else {
			bb3=(N3*oversamp-(N3-i3));
		}
		double correction3_re=zcorrection[bb3*2];
		double correction3_im=zcorrection[bb3*2+1];
		aa3=aa3*N1*N2;
		bb3=bb3*N1*oversamp*N2*oversamp;
		for (int i2=0; i2<N2; i2++) {
			int aa2=((i2+N2/2)%N2);
			int bb2=0;
			if (i2<(N2+1)/2) {
				bb2=i2;
			}
			else {
				bb2=(N2*oversamp-(N2-i2));
			}
			double correction2_re=correction3_re*ycorrection[bb2*2] - correction3_im*ycorrection[bb2*2+1];
			double correction2_im=correction3_re*ycorrection[bb2*2+1] + correction3_im*ycorrection[bb2*2];
			aa2=aa2*N1+aa3;
			bb2=bb2*N1*oversamp+bb3;
			for (int i1=0; i1<N1; i1++) {
				int aa1=(i1+N1/2)%N1;
				int bb1=0;
				if (i1<(N1+1)/2) {
					bb1=i1;
				}
				else {
					bb1=N1*oversamp-(N1-i1);
				}
				double correction1_re=correction2_re*xcorrection[bb1*2] - correction2_im*xcorrection[bb1*2+1];
				double correction1_im=correction2_re*xcorrection[bb1*2+1] + correction2_im*xcorrection[bb1*2];
				double correction1_magsqr=correction1_re*correction1_re+correction1_im*correction1_im;
				double correction1_inv_re=correction1_re/correction1_magsqr;
				double correction1_inv_im=-correction1_im/correction1_magsqr;
				aa1=aa1+aa2;
				bb1=bb1+bb2;
				out[aa1*2]=( out_oversamp_hat[bb1*2]*correction1_inv_re - out_oversamp_hat[bb1*2+1]*correction1_inv_im ) / M;
				out[aa1*2+1]=( out_oversamp_hat[bb1*2]*correction1_inv_im + out_oversamp_hat[bb1*2+1]*correction1_inv_re ) / M;
				//out[aa1*2]=out_oversamp_hat[bb1*2]; //for testing without correction
				//out[aa1*2+1]=out_oversamp_hat[bb1*2+1]; //for testing without correction
			}
		}
	}

	free(xcorrection);
	free(ycorrection);
	free(zcorrection);

#endif
}

bool check_valid_inputs(BlockSpread3DData &D) {
	for (int m=0; m<D.M; m++) {
		if ((D.x[m]<0)||(D.x[m]>=D.N1o)) {
			printf("Out of range: %g, %d\n",D.x[m],D.N1o);
			return false;
		}
		if ((D.y[m]<0)||(D.y[m]>=D.N2o)) {
			printf("Out of range: %g, %d\n",D.y[m],D.N2o);
			return false;
		}
		if ((D.z[m]<0)||(D.z[m]>=D.N3o)) {
			printf("Out of range: %g, %d\n",D.z[m],D.N3o);
			return false;
		}
	}

	printf("inputs are okay.\n");
	return true;
}



void do_spreading(BlockSpread3DData &D,const KernelInfo &KK1,const KernelInfo &KK2,const KernelInfo &KK3) {
	double x_kernel[KK1.nspread+1];
	double y_kernel[KK2.nspread+1];
	double z_kernel[KK3.nspread+1];

	double N1o_times_2=D.N1o*2;
	double N1oN2o_times_2=D.N1o*D.N2o*2;
	double N1oN2oN3o_times_2=D.N1o*D.N2o*D.N3o*2;
	for (int ii=0; ii<N1oN2oN3o_times_2; ii++) D.uniform_d[ii]=0;
	for (int jj=0; jj<D.M; jj++) {
		double x0=D.x[jj],y0=D.y[jj],z0=D.z[jj],d0_re=D.nonuniform_d[jj*2],d0_im=D.nonuniform_d[jj*2+1];

		int x_integer=ROUND_2_INT(x0);
		double x_diff=x0-x_integer;
		int y_integer=ROUND_2_INT(y0);
		double y_diff=y0-y_integer;
		int z_integer=ROUND_2_INT(z0);
		double z_diff=z0-z_integer;

		evaluate_kernel_1d(x_kernel,x_diff,-KK1.nspread/2,-KK1.nspread/2+KK1.nspread,KK1);
		evaluate_kernel_1d(y_kernel,y_diff,-KK2.nspread/2,-KK2.nspread/2+KK2.nspread,KK2);
		evaluate_kernel_1d(z_kernel,z_diff,-KK3.nspread/2,-KK3.nspread/2+KK3.nspread,KK3);

		int xmin,xmax;
		if (x_diff<0) {
			xmin=fmax(x_integer-KK1.nspread/2,0);
			xmax=fmin(x_integer+KK1.nspread/2-1,D.N1o-1);
		}
		else {
			xmin=fmax(x_integer-KK1.nspread/2+1,0);
			xmax=fmin(x_integer+KK1.nspread/2,D.N1o-1);
		}
		int iix=xmin-(x_integer-KK1.nspread/2);


		int ymin,ymax;
		if (y_diff<0) {
			ymin=fmax(y_integer-KK2.nspread/2,0);
			ymax=fmin(y_integer+KK2.nspread/2-1,D.N2o-1);
		}
		else {
			ymin=fmax(y_integer-KK2.nspread/2+1,0);
			ymax=fmin(y_integer+KK2.nspread/2,D.N2o-1);
		}
		int iiy=ymin-(y_integer-KK2.nspread/2);

		int zmin,zmax;
		if (z_diff<0) {
			zmin=fmax(z_integer-KK3.nspread/2,0);
			zmax=fmin(z_integer+KK3.nspread/2-1,D.N3o-1);
		}
		else {
			zmin=fmax(z_integer-KK3.nspread/2+1,0);
			zmax=fmin(z_integer+KK3.nspread/2,D.N3o-1);
		}
		int iiz=zmin-(z_integer-KK3.nspread/2);

		int xmax_minus_xmin_plus_iix=xmax-xmin+iix;
		int xmin_times_2=xmin*2;
		for (int iz=zmin; iz<=zmax; iz++) {
			double kernval_00=z_kernel[iz-zmin+iiz];
			int kkk1=N1oN2o_times_2*iz; //complex index
			int kkk2=kkk1+N1o_times_2*ymin; //remember complex
			for (int iy=ymin; iy<=ymax; iy++) {
				double kernval_01=kernval_00*y_kernel[iy-ymin+iiy];
				double kernval_01_times_d0_re=kernval_01*d0_re;
				double kernval_01_times_d0_im=kernval_01*d0_im;
				int kkk3=kkk2+xmin_times_2; //remember it is complex
				for (int ix0=iix; ix0<=xmax_minus_xmin_plus_iix; ix0++) {
					//most of the time is spent inside this inner-most loop
					D.uniform_d[kkk3]+=x_kernel[ix0]*kernval_01_times_d0_re;
					D.uniform_d[kkk3+1]+=x_kernel[ix0]*kernval_01_times_d0_im;
					kkk3+=2; //remember it is complex
				}
				kkk2+=N1o_times_2; //remember complex
			}
		}
	}
}

