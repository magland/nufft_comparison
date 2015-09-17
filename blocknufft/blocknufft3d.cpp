#include "blocknufft3d.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "qute.h"
#include "omp.h"
#include "fftw3.h"

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
	int nspread1,nspread2,nspread3; // Spreading diameters
	double tau1,tau2,tau3; //decay rates for convolution kernel
	double *x,*y,*z; // non-uniform locations
	double *nonuniform_d; // non-uniform data
	double *uniform_d; // uniform data
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

	// Transfer opts and other parameters to D
	BlockSpread3DData D;
	D.N1o=opts.N1o; D.N2o=opts.N2o; D.N3o=opts.N3o;
	D.nspread1=opts.nspread1; D.nspread2=opts.nspread2; D.nspread3=opts.nspread3;
	D.tau1=opts.tau1; D.tau2=opts.tau2; D.tau3=opts.tau3;
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
	do_spreading(D);
    printf("  --- Elapsed: %d ms\n",timer0.elapsed());

    printf("freeing data...\n"); timer0.start();

	free_data(D);
    printf("  --- Elapsed: %d ms\n",timer0.elapsed());

	return true;
}

// A couple implementation routines for nufft
bool do_fft_3d(int N1,int N2,int N3,double *out,double *in,int num_threads=1);
void do_fix_3d(int N1,int N2,int N3,int M,int oversamp,double tau,double *out,double *out_oversamp_hat);

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

// Here's the nufft!
bool blocknufft3d(const BlockNufft3DOptions &opts,double *out,double *spread,double *x,double *y,double *z,double *d) {
    QTime timer0;
    QTime timer_total; timer_total.start();

	printf("\nStarting blocknufft3d.\n");

    omp_set_num_threads(opts.num_threads);

	double eps=opts.eps;
	int oversamp=2; if (eps<= 1e-11) oversamp=3;
	int nspread=(int)(-log(eps)/(M_PI*(oversamp-1)/(oversamp-.5)) + .5) + 1; //the plus one was added -- different from docs -- aha!
	nspread=nspread*2; //we need to multiply by 2, because I consider nspread as the diameter
	double lambda=oversamp*oversamp * nspread/2 / (oversamp*(oversamp-.5));
	double tau=M_PI/lambda;
	printf("Using oversamp=%d, nspread=%d, tau=%g\n",oversamp,nspread,tau);

    //Initialize the pre-computed exponentials (do this outside a subthread!)
    exp_lookup_init(tau,tau,tau);

    int N1o=opts.N1*oversamp; int N2o=opts.N2*oversamp; int N3o=opts.N3*oversamp;

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
                    BD.N1o=BD.xmax-BD.xmin+1+nspread;
                    BD.N2o=BD.ymax-BD.ymin+1+nspread;
                    BD.N3o=BD.zmax-BD.zmin+1+nspread;
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
        BD->x[jj]=x[ii]-BD->xmin+nspread/2;
        BD->y[jj]=y[ii]-BD->ymin+nspread/2;
        BD->z[jj]=z[ii]-BD->zmin+nspread/2;
        BD->nonuniform_d[jj*2]=d[ii*2];
        BD->nonuniform_d[jj*2+1]=d[ii*2+1];
        BD->jj++;
    }
    printf("  --- Elapsed: %d ms\n",timer0.elapsed());

    printf("Spreading...\n"); timer0.start();
    #pragma omp parallel
    {
        if (omp_get_thread_num()==0) printf("#################### Using %d threads (%d prescribed)\n",omp_get_num_threads(),opts.num_threads);
        #pragma omp for
        for (int bb=0; bb<num_blocks; bb++) {
            BlockData *BD=&(block_data[bb]);
            BlockSpread3DOptions sopts;
            sopts.N1o=BD->N1o; sopts.N2o=BD->N2o; sopts.N3o=BD->N3o;
            sopts.M=BD->M;
            sopts.nspread1=nspread; sopts.nspread2=nspread; sopts.nspread3=nspread;
            sopts.tau1=tau; sopts.tau2=tau; sopts.tau3=tau;
            blockspread3d(sopts,BD->uniform_d,BD->x,BD->y,BD->z,BD->nonuniform_d);
        }
    }
    printf("  For blockspread3d: %d ms\n",timer0.elapsed());

    printf("Combining uniform data...\n"); timer0.start();
    int N1oN2oN3o=N1o*N2o*N3o;
    for (int ii=0; ii<N1oN2oN3o; ii++) {
        out_oversamp_hat[ii*2]=0;
        out_oversamp_hat[ii*2+1]=0;
    }
    for (int bb=0; bb<num_blocks; bb++) {
        BlockData *BD=&(block_data[bb]);
        int jjj=0;
        for (int j3=0; j3<BD->N3o; j3++) {
            int k3=(j3+BD->zmin-nspread/2+N3o)%N3o;
            int kkk3=k3*N1o*N2o;
            for (int j2=0; j2<BD->N2o; j2++) {
                int k2=(j2+BD->ymin-nspread/2+N2o)%N2o;
                int kkk2=kkk3+k2*N1o;
                for (int j1=0; j1<BD->N1o; j1++) {
                    int k1=(j1+BD->xmin-nspread/2+N1o)%N1o;
                    int kkk1=kkk2+k1;
                    out_oversamp[kkk1*2]+=BD->uniform_d[jjj*2];
                    out_oversamp[kkk1*2+1]+=BD->uniform_d[jjj*2+1];
                    jjj++;
                }
            }
        }
    }
    for (int ii=0; ii<N1oN2oN3o; ii++) {
        spread[ii*2]=out_oversamp[ii*2];
        spread[ii*2+1]=out_oversamp[ii*2+1];
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


    exp_lookup_destroy();

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




void do_spreading(BlockSpread3DData &D) {
    double x_term2[D.nspread1/2],x_term2_neg[D.nspread1/2];
    double y_term2[D.nspread2/2],y_term2_neg[D.nspread2/2];
    double z_term2[D.nspread3/2],z_term2_neg[D.nspread3/2];
    double precomp_x_term2[D.nspread1+1]; //conservatively large
    double N1o_times_2=D.N1o*2;
    double N1oN2o_times_2=D.N1o*D.N2o*2;
    double N1oN2oN3o_times_2=D.N1o*D.N2o*D.N3o*2;
    for (int ii=0; ii<N1oN2oN3o_times_2; ii++) D.uniform_d[ii]=0;
    for (int jj=0; jj<D.M; jj++) {
        double x0=D.x[jj],y0=D.y[jj],z0=D.z[jj],d0_re=D.nonuniform_d[jj*2],d0_im=D.nonuniform_d[jj*2+1];

        int x_integer=ROUND_2_INT(x0);
        double x_diff=x0-x_integer;
        double x_term1=exp(-x_diff*x_diff*D.tau1);
        double x_term2_factor=exp(2*x_diff*D.tau1);
        int xmin,xmax;
        if (x_diff<0) {
            xmin=fmax(x_integer-D.nspread1/2,0);
            xmax=fmin(x_integer+D.nspread1/2-1,D.N1o-1);
        }
        else {
            xmin=fmax(x_integer-D.nspread1/2+1,0);
            xmax=fmin(x_integer+D.nspread1/2,D.N1o-1);
        }

        int y_integer=ROUND_2_INT(y0);
        double y_diff=y0-y_integer;
        double y_term1=exp(-y_diff*y_diff*D.tau2);
        double y_term2_factor=exp(2*y_diff*D.tau2);
        int ymin,ymax;
        if (y_diff<0) {
            ymin=fmax(y_integer-D.nspread2/2,0);
            ymax=fmin(y_integer+D.nspread2/2-1,D.N2o-1);
        }
        else {
            ymin=fmax(y_integer-D.nspread2/2+1,0);
            ymax=fmin(y_integer+D.nspread2/2,D.N2o-1);
        }

        int z_integer=ROUND_2_INT(z0);
        double z_diff=z0-z_integer;
        double z_term1=exp(-z_diff*z_diff*D.tau3);
        double z_term2_factor=exp(2*z_diff*D.tau3);
        double zmin,zmax;
        if (z_diff<0) {
            zmin=fmax(z_integer-D.nspread3/2,0);
            zmax=fmin(z_integer+D.nspread3/2-1,D.N3o-1);
        }
        else {
            zmin=fmax(z_integer-D.nspread3/2+1,0);
            zmax=fmin(z_integer+D.nspread3/2,D.N3o-1);
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
            int kkk1=N1oN2o_times_2*iz; //complex index
            int iiz=iz-z_integer;
            double kernval1=kernval0;
            if (iiz>=0) kernval1*=z_term2[iiz];
            else kernval1*=z_term2_neg[-iiz];
            for (int iy=ymin; iy<=ymax; iy++) {
                int kkk2=kkk1+N1o_times_2*iy;
                int iiy=iy-y_integer;
                double kernval2=kernval1;
                if (iiy>=0) kernval2*=y_term2[iiy];
                else kernval2*=y_term2_neg[-iiy];
                int kkk3=kkk2+xmin*2;
                //did the precompute for for efficiency -- note, we don't need to check for negative
                for (int iii=0; iii<precomp_x_term2_sz; iii++) {
                    //most of the time is spent within this code block!!!
                    double tmp0=kernval2*precomp_x_term2[iii];
                    D.uniform_d[kkk3]+=d0_re*tmp0;
                    D.uniform_d[kkk3+1]+=d0_im*tmp0; //most of the time is spent on this line!!!
                    kkk3+=2; //plus two because complex
                }
            }
        }
    }
}

void free_data(BlockSpread3DData &D) {
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

