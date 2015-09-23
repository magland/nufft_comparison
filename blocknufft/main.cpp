#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "omp.h"
#include "blocknufft3d.h"
#include <chrono>
#include "qute.h"
#include "besseli.h"
#include <unistd.h>
#include <sys/mman.h>
#include <thread>

void test_openmp() {
	int N=1e6;
	int M=100;
	double X[N+M];
	for (int i=0; i<N+M; i++) X[i]=i;

	{
		QTime timer; timer.start();
		double val=0;
		for (int j=0; j<N; j++) {
			int sgn=1;
			for (int i=0; i<M; i++) {
				for (int k=0; k<M; k++) {
					val+=sgn*X[j]*X[j+i]*X[j+k];
					sgn*=-1;
				}
			}
		}
		printf("Result = %.0f\n",val);
		printf("Elapsed: %d ms\n",timer.elapsed());
	}

	{
		double val=0;
		QTime timer; timer.start();
		omp_set_num_threads(5);
		#pragma omp parallel
		{
			printf("# threads=%d\n",omp_get_num_threads());
			double local_val=0;
			QTime timerA; timerA.start();
			int num_handled_by_thread=0;
			int i1=omp_get_thread_num()*N/omp_get_num_threads();
			int i2=i1+N/omp_get_num_threads();
			if (omp_get_thread_num()==omp_get_num_threads()-1) {
				i2=N;
			}
			for (int j=i1; j<i2; j++) {
				num_handled_by_thread++;
				int sgn=1;
				for (int i=0; i<M; i++) {
					for (int k=0; k<M; k++) {
						//local_val+=sgn*X[j]*X[j+i]*X[j+k];
						local_val+=sgn*j*(j+i)*(j+k);
						sgn*=-1;
					}
				}
			}
			printf("Thread %d elapsed: %d, rate: %g for %d\n",omp_get_thread_num(),timerA.elapsed(),num_handled_by_thread*1.0/timerA.elapsed(),num_handled_by_thread);
			val+=local_val;
		}
		printf("Result = %.0f\n",val);
		printf("Elapsed: %d ms\n",timer.elapsed());
	}
}

double count_it_up(int j1,int j2,int M) {
	double val=0;
	QTime timerA; timerA.start();
	for (int j=j1; j<j2; j++) {
		int sgn=1;
		for (int i=0; i<M; i++) {
			for (int k=0; k<M; k++) {
				//local_val+=sgn*X[j]*X[j+i]*X[j+k];
				val+=sgn*j*(j+i)*(j+k);
				sgn*=-1;
			}
		}
	}
	printf("Elapsed time for process: %d, rate: %g\n",timerA.elapsed(),(j2-j1)*1.0/timerA.elapsed());
	return val;
}

void test_fork() {
	int N=1e6;
	int M=100;
	int num_processes=2;

	int *number=(int *)mmap(NULL, sizeof(int), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);

	pid_t pid=fork();
	if (pid==0) {
		*number=100;
		int i=0;
		int j1=i*N/num_processes;
		int j2=(i+1)*N/num_processes;
		if (i+1==num_processes) j2=N;
		double val1=count_it_up(j1,j2,M);
		printf("The child, %.0f!\n",val1);
	}
	else if (pid>0) {
		int i=1;
		int j1=i*N/num_processes;
		int j2=(i+1)*N/num_processes;
		if (i+1==num_processes) j2=N;
		double val1=count_it_up(j1,j2,M);
		printf("The parent, %.0f, number = %d!\n",val1,*number);
	}
	else {
		printf("Fork failed!\n");
	}
}

void count_it_up_2(int j1,int j2,int M) {
	double val=0;
	QTime timerA; timerA.start();
	for (int j=j1; j<j2; j++) {
		int sgn=1;
		for (int i=0; i<M; i++) {
			for (int k=0; k<M; k++) {
				//local_val+=sgn*X[j]*X[j+i]*X[j+k];
				val+=sgn*j*(j+i)*(j+k);
				sgn*=-1;
			}
		}
	}
	printf("Elapsed time for thread (%d,%d,%d): %d, rate: %g, val: %.0f\n",j1,j2,M,timerA.elapsed(),(j2-j1)*1.0/timerA.elapsed(),val);
}

/*
void test_thread() {
	int N=1e6;
	int M=100;
	int num_threads=5;

	QList<std::thread *> threads;
	for (int i=0; i<num_threads; i++) {
		int j1=i*N/num_threads;
		int j2=(i+1)*N/num_threads;
		std::thread *T1=new std::thread(count_it_up_2,j1,j2,M);
		threads << T1;
	}

	for (int i=0; i<num_threads; i++) {
		threads[i]->join();
	}

	printf("Okay!\n");
}
*/

void test_list() {
	QList<int> list;
	printf("test_list\n");
	for (int i=0; i<20; i++) {
		list.append(i);
	}
	for (int i=0; i<list.count(); i++) {
		printf("%d: %d\n",i,list.value(i));
	}
}

int main(int argc, char *argv[])
{

	//test_openmp();
	//test_qthread();
	//test_fork();
	//test_thread();
	//test_list();
	//return 0;

	/*
	 //Time bessel evaluation
	printf("Timing bessel evaluations...\n");
	QTime timer; timer.start();
	double tmp=0;
	for (int i=0; i<1e8; i++) {
		tmp+=besseli0(0.3+i*1.0/1e8);

	}
	printf("%.10f\n",tmp);
	printf("Elapsed: %d ms\n",timer.elapsed());
	*/

    int N1=200;
    int N2=200;
    int N3=200;
	int M=200*200*200;
    double *uniform_d=(double *)malloc(sizeof(double)*N1*N2*N3*2); //the output (complex)
    double *xyz=(double *)malloc(sizeof(double)*M*3); //the input sample locations
    double *nonuniform_d=(double *)malloc(sizeof(double)*M*2); //the input data values (complex)
	double eps=1e-3; //this will determine the spreading kernel size! For gaussian, 1e-3 gives 8, 1e-6 gives 16, I believe
	int K1=80,K2=80,K3=80; //the block size for blocking. You could set these to 10000 to get just a single block.
	int num_threads=6;
	int kernel_type=KERNEL_TYPE_KB;

    int cc=0;
    for (int aa=0; aa<3; aa++) {
        for (int ii=0; ii<M; ii++) {
            xyz[cc]=rand()*1.0/RAND_MAX*2*M_PI;
            cc++;
        }
    }

    for (int ii=0; ii<M; ii++) {
        nonuniform_d[ii]=rand()*1.0/RAND_MAX * 2 - 1;
    }

    //here is the procedure...
    printf("Running blocknufft3d...\n");
	blocknufft3d(N1,N2,N3,M,uniform_d,xyz,nonuniform_d,eps,K1,K2,K3,num_threads,kernel_type);

    free(uniform_d);
    free(xyz);
    free(nonuniform_d);

	return 0;
}
