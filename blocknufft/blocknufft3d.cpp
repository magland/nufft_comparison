#include "blocknufft3d.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "qute.h"
#include "omp.h"
#include "fftw3.h"

//For the location codes
#define LEFT_SIDE 1
#define RIGHT_SIDE 2
#define BOTTOM_SIDE 4
#define TOP_SIDE 8
#define BACK_SIDE 16
#define FRONT_SIDE 32

// The following is needed because incredibly C++ does not seem to have a round() function
#define ROUND_2_INT(f) ((int)(f >= 0.0 ? (f + 0.5) : (f - 0.5)))

// These are the the pre-computed exponentials (static variables)
double *s_exp_lookup1=0;
double *s_exp_lookup2=0;
double *s_exp_lookup3=0;
void exp_lookup_init(double tau1,double tau2,double tau3);
void exp_lookup_destroy();

// This is the data needed for the spreading
struct BlockSpread3DData {
	int M; // Number of non-uniform points
	int N1o,N2o,N3o; // Oversampled dimensions
	int K1,K2,K3; // Block dimensions
	int nspread1,nspread2,nspread3; // Spreading diameters
	double tau1,tau2,tau3; //decay rates for convolution kernel
	int num_threads; // number of parallel threads
	double *x,*y,*z; // non-uniform locations
	double *nonuniform_d; // non-uniform data
	double *uniform_d; // uniform data

	int num_blocks_x,num_blocks_y,num_blocks_z; // The number of blocks in each of the three directions
	int num_blocks; // Total number of blocks -- the product of num_blocks_x, num_blocks_y, and num_blocks_z
	int *location_codes; // (length=M) These are internal codes which are used to indicate which
	int *block_ids; // (length=M) we identify which blocks each non-uniform point is in
	int *nonuniform_block_counts; // (length=num_blocks) The number of non-uniform points in each block
	int working_nonuniform_size; // The size of the working non-uniform data (which includes redundancies near the interfaces between adjacent blocks)
	int *nonuniform_block_indices; // (length=num_blocks) The indices in the working_x,working_y,working_z,working_d where each block starts
	int working_uniform_size; // The size of the working uniform data
	int *uniform_block_indices; // (length=num_blocks) The indices in the working_uniform where each block starts
	double *working_uniform_d; // (length=working_uniform_size)
	double *working_x,*working_y,*working_z,*working_d; // The working non-uniform data, includes redundancies near interfaces between adjacent blocks
};

// These are the implementation routines for spreading
bool check_valid_inputs(BlockSpread3DData &D);
void define_block_ids_and_location_codes(BlockSpread3DData &D);
void compute_nonuniform_block_counts(BlockSpread3DData &D);
void compute_sizes_and_block_indices(BlockSpread3DData &D);
void set_working_nonuniform_data(BlockSpread3DData &D);
void do_spreading(BlockSpread3DData &D);
void set_uniform_data(BlockSpread3DData &D);
void free_data(BlockSpread3DData &D);

// Here's the spreading!
bool blockspread3d(const BlockSpread3DOptions &opts,double *uniform_d,double *x,double *y,double *z,double *nonuniform_d) {
	QTime timer0;
	printf("Starting blockspread3d...\n");

	//Initialize the pre-computed exponentials
    exp_lookup_init(opts.tau1,opts.tau2,opts.tau3);
	//Set the number of parallel threads
	omp_set_num_threads(opts.num_threads);

	// Transfer opts and other parameters to D
	BlockSpread3DData D;
	D.N1o=opts.N1o; D.N2o=opts.N2o; D.N3o=opts.N3o;
	D.K1=opts.K1; D.K2=opts.K2; D.K3=opts.K3;
	D.nspread1=opts.nspread1; D.nspread2=opts.nspread2; D.nspread3=opts.nspread3;
	D.tau1=opts.tau1; D.tau2=opts.tau2; D.tau3=opts.tau3;
	D.num_threads=opts.num_threads;
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

	//Compute the number of blocks
	D.num_blocks_x=ceil(D.N1o*1.0/D.K1);
	D.num_blocks_y=ceil(D.N2o*1.0/D.K2);
	D.num_blocks_z=ceil(D.N3o*1.0/D.K3);
	D.num_blocks=D.num_blocks_x*D.num_blocks_y*D.num_blocks_z;

	printf("Defining block ids and location codes... "); timer0.start();
	D.location_codes=(int *)malloc(sizeof(int)*D.M);
	D.block_ids=(int *)malloc(sizeof(int)*D.M);
	define_block_ids_and_location_codes(D);
	printf("Elapsed: %d ms\n",timer0.elapsed());

	printf("Computing nonuniform block counts... "); timer0.start();
	compute_nonuniform_block_counts(D);
	printf("Elapsed: %d ms\n",timer0.elapsed());

	printf("Computing sizes and block indices... "); timer0.start();
	compute_sizes_and_block_indices(D);
	printf("Elapsed: %d ms\n",timer0.elapsed());

	printf("Initializing working uniform data... "); timer0.start();
	D.working_uniform_d=(double *)malloc(sizeof(double)*D.working_uniform_size*2); //complex
    #pragma omp parallel
    {
        #pragma omp for
		for (int ii=0; ii<D.working_uniform_size*2; ii++) D.working_uniform_d[ii]=0;
    }
	printf("Elapsed: %d ms\n",timer0.elapsed());

	printf("setting working nonuniform data... "); timer0.start();
	set_working_nonuniform_data(D);
	printf("Elapsed: %d ms\n",timer0.elapsed());

    printf("spreading []... "); timer0.start();
	do_spreading(D);
	printf("Elapsed: %d ms\n",timer0.elapsed());

	printf("setting uniform data... "); timer0.start();
	set_uniform_data(D);
	printf("Elapsed: %d ms\n",timer0.elapsed());

	printf("freeing data... "); timer0.start();
	exp_lookup_destroy();
	free_data(D);
	printf("Elapsed: %d ms\n",timer0.elapsed());

	return true;
}

// A couple implementation routines for nufft
bool do_fft_3d(int N1,int N2,int N3,double *out,double *in,int num_threads=1);
void do_fix_3d(int N1,int N2,int N3,int M,int oversamp,double tau,double *out,double *out_oversamp_hat);

// Here's the nufft!
bool blocknufft3d(const BlockNufft3DOptions &opts,double *out,double *spread,double *x,double *y,double *z,double *d) {
	QTime timer0; QTime timer_total; timer_total.start();

	printf("\nStarting blocknufft3d.\n");

	double eps=opts.eps;
	int oversamp=2; if (eps<= 1e-11) oversamp=3;
	int nspread=(int)(-log(eps)/(M_PI*(oversamp-1)/(oversamp-.5)) + .5) + 1; //the plus one was added -- different from docs -- aha!
	nspread=nspread*2; //we need to multiply by 2, because I consider nspread as the diameter
	double lambda=oversamp*oversamp * nspread/2 / (oversamp*(oversamp-.5));
	double tau=M_PI/lambda;
	printf("Using oversamp=%d, nspread=%d, tau=%g\n",oversamp,nspread,tau);

	BlockSpread3DOptions sopts;
	sopts.N1o=opts.N1*oversamp; sopts.N2o=opts.N2*oversamp; sopts.N3o=opts.N3*oversamp;
	sopts.K1=opts.K1; sopts.K2=opts.K2; sopts.K3=opts.K3;
	sopts.M=opts.M;
	sopts.num_threads=opts.num_threads;
	sopts.nspread1=nspread; sopts.nspread2=nspread; sopts.nspread3=nspread;
	sopts.tau1=tau; sopts.tau2=tau; sopts.tau3=tau;

	printf("Allocating...\n"); timer0.start();
	double *out_oversamp=(double *)malloc(sizeof(double)*sopts.N1o*sopts.N2o*sopts.N3o*2);
	double *out_oversamp_hat=(double *)malloc(sizeof(double)*sopts.N1o*sopts.N2o*sopts.N3o*2);
	printf("  --- Elapsed: %d ms\n",timer0.elapsed());

	printf("Scaling coordinates...\n"); timer0.start();
	double factor_x=sopts.N1o/(2*M_PI);
	double factor_y=sopts.N2o/(2*M_PI);
	double factor_z=sopts.N3o/(2*M_PI);
	for (int ii=0; ii<opts.M; ii++) {
		x[ii]*=factor_x;
		y[ii]*=factor_y;
		z[ii]*=factor_z;
	}
	printf("  --- Elapsed: %d ms\n",timer0.elapsed());

	printf("Spreading...\n"); timer0.start();
	blockspread3d(sopts,out_oversamp,x,y,z,d);
	double spreading_time=timer0.elapsed();
	printf("  --- Total time for spreading: %d ms\n",timer0.elapsed());
	for (int jj=0; jj<sopts.N1o*sopts.N2o*sopts.N3o*2; jj++) {
		spread[jj]=out_oversamp[jj];
	}

	printf("fft...\n"); timer0.start();
	if (!do_fft_3d(sopts.N1o,sopts.N2o,sopts.N3o,out_oversamp_hat,out_oversamp,opts.num_threads)) {
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
	do_fix_3d(opts.N1,opts.N2,opts.N3,opts.M,oversamp,tau,out,out_oversamp_hat);
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
	printf("   %.3f spreading, %.3f fft, %.3f other\n",spreading_time/1000,fft_time/1000,other_time/1000);
	printf("   %.1f%% spreading, %.1f%% fft, %.1f%% other\n",spreading_time/total_time*100,fft_time/total_time*100,other_time/total_time*100);

	printf("done with blocknufft3d.\n");
	return true;
}

// This is the mcwrap interface
void blocknufft3d(int N1,int N2,int N3,int M,double *uniform_d,double *spread,double *xyz,double *nonuniform_d,double eps,int K1,int K2,int K3,int num_threads) {
	BlockNufft3DOptions opts;
	opts.eps=eps;
	opts.K1=K1; opts.K2=K2; opts.K3=K3;
	opts.N1=N1; opts.N2=N2; opts.N3=N3;
	opts.M=M;
	opts.num_threads=num_threads;

	double *x=&xyz[0];
	double *y=&xyz[M];
	double *z=&xyz[2*M];
	blocknufft3d(opts,uniform_d,spread,x,y,z,nonuniform_d);
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
void do_fix_3d(int N1,int N2,int N3,int M,int oversamp,double tau,double *out,double *out_oversamp_hat) {
    double *correction_vals1=(double *)malloc(sizeof(double)*(N1/2+2));
    double *correction_vals2=(double *)malloc(sizeof(double)*(N2/2+2));
	double *correction_vals3=(double *)malloc(sizeof(double)*(N3/2+2));
	double lambda=M_PI/tau;
	int N1b=N1*oversamp;
	int N2b=N2*oversamp;
	int N3b=N3*oversamp;
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
}

bool check_valid_inputs(BlockSpread3DData &D) {
	if ((D.nspread1>D.K1)||(D.nspread2>D.K2)||(D.nspread3>D.K3)) {
		printf("nspread cannot be bigger than block size\n");
		return false;
	}
	bool ok=true; //we are doing a parallel check for valid inputs, so we need to do it this way
	#pragma omp parallel
	{
		bool local_ok=true;
		#pragma omp for
		for (int m=0; m<D.M; m++) {
			if ((D.x[m]<0)||(D.x[m]>=D.N1o)) {
				local_ok=false;
			}
			if ((D.y[m]<0)||(D.y[m]>=D.N2o)) {
				local_ok=false;
			}
			if ((D.z[m]<0)||(D.z[m]>=D.N3o)) {
				local_ok=false;
			}
		}
		#pragma omp critical
		{
			if (!local_ok) ok=false; //this is the way that the threads will not get locked. Just set the ok variable once for each thread.
		}
	}
	if (!ok) {
		printf("Problem in blockspread3d: Something is out of range.\n");
		return false;
	}

	return true;
}

void define_block_ids_and_location_codes(BlockSpread3DData &D) {
	#pragma omp parallel
	{
		#pragma omp for
		for (int m=0; m<D.M; m++) {
			int b1=(int)(D.x[m]/D.K1);
			int c1=ROUND_2_INT(D.x[m]-b1*D.K1);
			int b2=(int)(D.y[m]/D.K2);
			int c2=ROUND_2_INT(D.y[m]-b2*D.K2);
			int b3=(int)(D.z[m]/D.K3);
			int c3=ROUND_2_INT(D.z[m]-b3*D.K3);
			int code=0;
			int k1=D.K1,k2=D.K2,k3=D.K3;
			if (b1==D.num_blocks_x-1) k1=fmin(D.N1o-(D.num_blocks_x-1)*D.K1,D.K1); //the min is needed in case K1>N1*2
			if (b2==D.num_blocks_y-1) k2=fmin(D.N2o-(D.num_blocks_y-1)*D.K2,D.K2);
			if (b3==D.num_blocks_z-1) k3=fmin(D.N3o-(D.num_blocks_z-1)*D.K3,D.K3);
			if ((k1-c1<=D.nspread1/2)) code=code|RIGHT_SIDE;
			if ((c1<=D.nspread1/2)) code=code|LEFT_SIDE;
			if ((k2-c2<=D.nspread2/2)) code=code|TOP_SIDE;
			if ((c2<=D.nspread2/2)) code=code|BOTTOM_SIDE;
			if ((k3-c3<=D.nspread3/2)) code=code|FRONT_SIDE;
			if ((c3<=D.nspread3)) code=code|BACK_SIDE;
			D.location_codes[m]=code;
			D.block_ids[m]=b1+D.num_blocks_x*b2+D.num_blocks_x*D.num_blocks_y*b3;
		}
	}
}

void compute_nonuniform_block_counts(BlockSpread3DData &D) {
	D.nonuniform_block_counts=(int *)malloc(sizeof(int)*D.num_blocks);
	for (int i=0; i<D.num_blocks; i++) D.nonuniform_block_counts[i]=0;
	#pragma omp parallel
	{
		int local_nonuniform_block_counts[D.num_blocks]; for (int i=0; i<D.num_blocks; i++) local_nonuniform_block_counts[i]=0;
		#pragma omp for
		for (int m=0; m<D.M; m++) {
			int code=D.location_codes[m];
			int bb0=D.block_ids[m];
			int bx=bb0%D.num_blocks_x;
			int by=(bb0/D.num_blocks_x)%D.num_blocks_y;
			int bz=(bb0/D.num_blocks_x/D.num_blocks_y);
			for (int i1=-1; i1<=1; i1++) {
				if ( ((i1==-1)&&(code&LEFT_SIDE)) || (i1==0) || ((i1==1)&&(code&RIGHT_SIDE)) ) {
					int bbx=bx+i1;
					if (bbx<0) {bbx+=D.num_blocks_x;}
					if (bbx>=D.num_blocks_x) {bbx-=D.num_blocks_x;}
					for (int i2=-1; i2<=1; i2++) {
						if ( ((i2==-1)&&(code&BOTTOM_SIDE)) || (i2==0) || ((i2==1)&&(code&TOP_SIDE)) ) {
							int bby=by+i2;
							if (bby<0) {bby+=D.num_blocks_y;}
							if (bby>=D.num_blocks_y) {bby-=D.num_blocks_y;}
							for (int i3=-1; i3<=1; i3++) {
								if ( ((i3==-1)&&(code&BACK_SIDE)) || (i3==0) || ((i3==1)&&(code&FRONT_SIDE)) ) {
									int bbz=bz+i3;
									if (bbz<0) {bbz+=D.num_blocks_z;}
									if (bbz>=D.num_blocks_z) {bbz-=D.num_blocks_z;}
									int bbb=bbx+D.num_blocks_x*bby+D.num_blocks_x*D.num_blocks_y*bbz;
									local_nonuniform_block_counts[bbb]++;
								}
							}
						}
					}
				}
			}
		}
		#pragma omp critical
		{
			for (int i=0; i<D.num_blocks; i++) D.nonuniform_block_counts[i]+=local_nonuniform_block_counts[i];
		}
	}
}

void compute_sizes_and_block_indices(BlockSpread3DData &D) {
	D.working_nonuniform_size=0;
	D.nonuniform_block_indices=(int *)malloc(sizeof(int)*D.num_blocks);
	D.working_uniform_size=0;
	D.uniform_block_indices=(int *)malloc(sizeof(int)*D.num_blocks);
	for (int i=0; i<D.num_blocks; i++) {
		int b1=i%D.num_blocks_x;
		int b2=(i/D.num_blocks_x)%D.num_blocks_y;
		int b3=(i/D.num_blocks_x/D.num_blocks_y);
		D.nonuniform_block_indices[i]=D.working_nonuniform_size;
		D.working_nonuniform_size+=D.nonuniform_block_counts[i];
		D.uniform_block_indices[i]=D.working_uniform_size;
		int F1=fmin(D.K1,D.N1o-b1*D.K1);
		int F2=fmin(D.K2,D.N2o-b2*D.K2);
		int F3=fmin(D.K3,D.N3o-b3*D.K3);
		D.working_uniform_size+=F1*F2*F3;
	}
}

void set_working_nonuniform_data(BlockSpread3DData &D) {
	int block_ii[D.num_blocks]; for (int i=0; i<D.num_blocks; i++) block_ii[i]=0;
	D.working_x=(double *)malloc(sizeof(double)*D.working_nonuniform_size);
	D.working_y=(double *)malloc(sizeof(double)*D.working_nonuniform_size);
	D.working_z=(double *)malloc(sizeof(double)*D.working_nonuniform_size);
	D.working_d=(double *)malloc(sizeof(double)*D.working_nonuniform_size*2); //times 2 because complex
	for (int m=0; m<D.M; m++) { //can this be parallelized? Not sure!
		int code=D.location_codes[m];
		int bb0=D.block_ids[m];
		int bx=bb0%D.num_blocks_x;
		int by=(bb0/D.num_blocks_x)%D.num_blocks_y;
		int bz=(bb0/D.num_blocks_x/D.num_blocks_y);
		for (int i1=-1; i1<=1; i1++) {
			if ( ((i1==-1)&&(code&LEFT_SIDE)) || (i1==0) || ((i1==1)&&(code&RIGHT_SIDE)) ) {
				int bbx=bx+i1;
				int wrapx=0;
				if (bbx<0) {bbx+=D.num_blocks_x; wrapx=D.N1o;}
				if (bbx>=D.num_blocks_x) {bbx-=D.num_blocks_x; wrapx=-D.N1o;}
				for (int i2=-1; i2<=1; i2++) {
					if ( ((i2==-1)&&(code&BOTTOM_SIDE)) || (i2==0) || ((i2==1)&&(code&TOP_SIDE)) ) {
						int bby=by+i2;
						int wrapy=0;
						if (bby<0) {bby+=D.num_blocks_y; wrapy=D.N2o;}
						if (bby>=D.num_blocks_y) {bby-=D.num_blocks_y; wrapy=-D.N2o;}
						for (int i3=-1; i3<=1; i3++) {
							if ( ((i3==-1)&&(code&BACK_SIDE)) || (i3==0) || ((i3==1)&&(code&FRONT_SIDE)) ) {
								int bbz=bz+i3;
								int wrapz=0;
								if (bbz<0) {bbz+=D.num_blocks_z; wrapz=D.N3o;}
								if (bbz>=D.num_blocks_z) {bbz-=D.num_blocks_z; wrapz=-D.N3o;}
								int bbb=bbx+D.num_blocks_x*bby+D.num_blocks_x*D.num_blocks_y*bbz;
								int iii=D.nonuniform_block_indices[bbb]+block_ii[bbb];
								D.working_x[iii]=D.x[m]+wrapx;
								D.working_y[iii]=D.y[m]+wrapy;
								D.working_z[iii]=D.z[m]+wrapz;
								D.working_d[iii*2]=D.nonuniform_d[m*2];
								D.working_d[iii*2+1]=D.nonuniform_d[m*2+1];
								block_ii[bbb]++;
							}
						}
					}
				}
			}
		}
	}
}

void do_spreading(BlockSpread3DData &D) {
	#pragma omp parallel
	{
		if (omp_get_thread_num()==0) printf("#################### Using %d threads (%d prescribed)\n",omp_get_num_threads(),D.num_threads);
		#pragma omp for
		for (int iblock=0; iblock<D.num_blocks; iblock++) {
			int cc1=iblock%D.num_blocks_x;
			int cc2=(iblock%(D.num_blocks_x*D.num_blocks_y))/D.num_blocks_x;
			int cc3=iblock/(D.num_blocks_x*D.num_blocks_y);
			int factor1=D.K1; if ((cc1+1)*D.K1>=D.N1o) factor1=D.N1o-cc1*D.K1;
			int factor2=D.K2; if ((cc2+1)*D.K2>=D.N2o) factor2=D.N2o-cc2*D.K2;
			//int factor3=K3; if ((cc3+1)*K3>=N3) factor3=N3-cc3*K3;
			int factor12=factor1*factor2;
			int factor1_times_2=factor1*2; int factor12_times_2=factor12*2;
			int block_xmin=cc1*D.K1,block_xmax=(cc1+1)*D.K1-1; if (block_xmax>=D.N1o) block_xmax=D.N1o-1;
			int block_ymin=cc2*D.K2,block_ymax=(cc2+1)*D.K2-1; if (block_ymax>=D.N2o) block_ymax=D.N2o-1;
			int block_zmin=cc3*D.K3,block_zmax=(cc3+1)*D.K3-1; if (block_zmax>=D.N3o) block_zmax=D.N3o-1;
			int jj=D.nonuniform_block_indices[iblock];
			int tmp=jj+D.nonuniform_block_counts[iblock];
			double x_term2[D.nspread1/2],x_term2_neg[D.nspread1/2];
			double y_term2[D.nspread2/2],y_term2_neg[D.nspread2/2];
			double z_term2[D.nspread3/2],z_term2_neg[D.nspread3/2];
			double precomp_x_term2[D.nspread1+1]; //conservatively large
			while (jj<tmp) {
				double x0=D.working_x[jj],y0=D.working_y[jj],z0=D.working_z[jj],d0_re=D.working_d[jj*2],d0_im=D.working_d[jj*2+1];

				//exp(-(i+d-j)^2*tau) = exp(-(i-j)^2*tau)*exp(-j^2*tau)*exp(-2*(i-j)*dd*tau)

				int x_integer=ROUND_2_INT(x0);
				double x_diff=x0-x_integer;
				double x_term1=exp(-x_diff*x_diff*D.tau1);
				double x_term2_factor=exp(2*x_diff*D.tau1);
				int xmin,xmax;
				if (x_diff<0) {
					xmin=fmax(x_integer-D.nspread1/2,block_xmin);
					xmax=fmin(x_integer+D.nspread1/2-1,block_xmax);
				}
				else {
					xmin=fmax(x_integer-D.nspread1/2+1,block_xmin);
					xmax=fmin(x_integer+D.nspread1/2,block_xmax);
				}

				int y_integer=ROUND_2_INT(y0);
				double y_diff=y0-y_integer;
				double y_term1=exp(-y_diff*y_diff*D.tau2);
				double y_term2_factor=exp(2*y_diff*D.tau2);
				int ymin,ymax;
				if (y_diff<0) {
					ymin=fmax(y_integer-D.nspread2/2,block_ymin);
					ymax=fmin(y_integer+D.nspread2/2-1,block_ymax);
				}
				else {
					ymin=fmax(y_integer-D.nspread2/2+1,block_ymin);
					ymax=fmin(y_integer+D.nspread2/2,block_ymax);
				}

				int z_integer=ROUND_2_INT(z0);
				double z_diff=z0-z_integer;
				double z_term1=exp(-z_diff*z_diff*D.tau3);
				double z_term2_factor=exp(2*z_diff*D.tau3);
				double zmin,zmax;
				if (z_diff<0) {
					zmin=fmax(z_integer-D.nspread3/2,block_zmin);
					zmax=fmin(z_integer+D.nspread3/2-1,block_zmax);
				}
				else {
					zmin=fmax(z_integer-D.nspread3/2+1,block_zmin);
					zmax=fmin(z_integer+D.nspread3/2,block_zmax);
				}

				x_term2[0]=1;
				x_term2_neg[0]=1;
				int aamax=D.nspread1/2;
				for (int aa=1; aa<=aamax; aa++) {
					x_term2[aa]=x_term2[aa-1]*x_term2_factor;
					x_term2_neg[aa]=x_term2_neg[aa-1]/x_term2_factor;
				}
				y_term2[0]=1;
				y_term2_neg[0]=1;
				int bbmax=D.nspread2/2;
				for (int bb=1; bb<=bbmax; bb++) {
					y_term2[bb]=y_term2[bb-1]*y_term2_factor;
					y_term2_neg[bb]=y_term2_neg[bb-1]/y_term2_factor;
				}
				z_term2[0]=1;
				z_term2_neg[0]=1;
				int ccmax=D.nspread3/2;
				for (int cc=1; cc<=ccmax; cc++) {
					z_term2[cc]=z_term2[cc-1]*z_term2_factor;
					z_term2_neg[cc]=z_term2_neg[cc-1]/z_term2_factor;
				}

				for (int aa=0; aa<=aamax; aa++) {
					x_term2[aa]*=s_exp_lookup1[aa];
					x_term2_neg[aa]*=s_exp_lookup1[aa];
				}
				for (int bb=0; bb<=bbmax; bb++) {
					y_term2[bb]*=s_exp_lookup2[bb];
					y_term2_neg[bb]*=s_exp_lookup2[bb];
				}
				for (int cc=0; cc<=ccmax; cc++) {
					z_term2[cc]*=s_exp_lookup3[cc];
					z_term2_neg[cc]*=s_exp_lookup3[cc];
				}

				int precomp_x_term2_sz=xmax-xmin+1;
				for (int ix=xmin; ix<=xmax; ix++) {
					int iix=ix-x_integer;
					if (iix>=0) precomp_x_term2[ix-xmin]=x_term2[iix];
					else precomp_x_term2[ix-xmin]=x_term2_neg[-iix];
				}

				double kernval0=x_term1*y_term1*z_term1;
				for (int iz=zmin; iz<=zmax; iz++) {
					int kkk1=D.uniform_block_indices[iblock]*2+factor12_times_2*(iz-block_zmin); //complex index
					int iiz=iz-z_integer;
					double kernval1=kernval0;
					if (iiz>=0) kernval1*=z_term2[iiz];
					else kernval1*=z_term2_neg[-iiz];
					for (int iy=ymin; iy<=ymax; iy++) {
						int kkk2=kkk1+factor1_times_2*(iy-block_ymin);
						int iiy=iy-y_integer;
						double kernval2=kernval1;
						if (iiy>=0) kernval2*=y_term2[iiy];
						else kernval2*=y_term2_neg[-iiy];
						int kkk3=kkk2+(xmin-block_xmin)*2; //times 2 because complex
						//did the precompute for for efficiency -- note, we don't need to check for negative
						for (int iii=0; iii<precomp_x_term2_sz; iii++) {
							//most of the time is spent within this code block!!!
							double tmp0=kernval2*precomp_x_term2[iii];
							D.working_uniform_d[kkk3]+=d0_re*tmp0;
							D.working_uniform_d[kkk3+1]+=d0_im*tmp0; //most of the time is spent on this line!!!
							kkk3+=2; //plus two because complex
						}
					}
				}

				//printf("%d,%d,%d,%d  %d,%d,%d,%d  %d,%d,%d,%d  %d\n",xmin,xmax,block_xmin,block_xmax,ymin,ymax,block_ymin,block_ymax,zmin,zmax,block_zmin,block_zmax,debug_ct);
				jj++;
			}
		}
	}
}

void set_uniform_data(BlockSpread3DData &D) {
	#pragma omp parallel
	{
		#pragma omp for
		for (int iblock=0; iblock<D.num_blocks; iblock++) {
			int cc1=iblock%D.num_blocks_x;
			int cc2=(iblock%(D.num_blocks_x*D.num_blocks_y))/D.num_blocks_x;
			int cc3=iblock/(D.num_blocks_x*D.num_blocks_y);
			int factor1=D.K1; if ((cc1+1)*D.K1>=D.N1o) factor1=D.N1o-cc1*D.K1;
			int factor2=D.K2; if ((cc2+1)*D.K2>=D.N2o) factor2=D.N2o-cc2*D.K2;
			int factor3=D.K3; if ((cc3+1)*D.K3>=D.N3o) factor3=D.N3o-cc3*D.K3;
			int dd1=cc1*D.K1;
			int dd2=cc2*D.K2;
			int dd3=cc3*D.K3;
			int kkk=D.uniform_block_indices[iblock]*2; //times 2 because complex
			for (int i3=0; i3<factor3; i3++) {
				for (int i2=0; i2<factor2; i2++) {
					for (int i1=0; i1<factor1; i1++) { //make this inner loop more efficient by not doing the multiplication here?
						int jjj=(dd1+i1)+(dd2+i2)*D.N1o+(dd3+i3)*D.N1o*D.N2o;
						D.uniform_d[jjj*2]=D.working_uniform_d[kkk];
						D.uniform_d[jjj*2+1]=D.working_uniform_d[kkk+1];
						kkk+=2; //add 2 because complex
					}
				}
			}
		}
	}
}

void free_data(BlockSpread3DData &D) {
	free(D.location_codes);
	free(D.block_ids);
	free(D.nonuniform_block_counts);
	free(D.nonuniform_block_indices);
	free(D.uniform_block_indices);
	free(D.working_uniform_d);
	free(D.working_x);
	free(D.working_y);
	free(D.working_z);
	free(D.working_d);
}

void exp_lookup_init(double tau1,double tau2,double tau3) {
	if (s_exp_lookup1) free(s_exp_lookup1);
	if (s_exp_lookup2) free(s_exp_lookup2);
	if (s_exp_lookup3) free(s_exp_lookup3);
	int NN=100;
	s_exp_lookup1=(double *)malloc(sizeof(double)*NN);
	s_exp_lookup2=(double *)malloc(sizeof(double)*NN);
	s_exp_lookup3=(double *)malloc(sizeof(double)*NN);
	for (int k0=0; k0<NN; k0++) {
		//s_exp_lookup1[k0]=exp(-k0*k0/(4*tau1));
		//s_exp_lookup2[k0]=exp(-k0*k0/(4*tau2));
		//s_exp_lookup3[k0]=exp(-k0*k0/(4*tau3));
		s_exp_lookup1[k0]=exp(-k0*k0*tau1);
		s_exp_lookup2[k0]=exp(-k0*k0*tau2);
		s_exp_lookup3[k0]=exp(-k0*k0*tau3);
	}
}

void exp_lookup_destroy() {
	free(s_exp_lookup1); s_exp_lookup1=0;
	free(s_exp_lookup2); s_exp_lookup2=0;
	free(s_exp_lookup3); s_exp_lookup3=0;
}

