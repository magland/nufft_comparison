/******************************************************
** See the accompanying README and LICENSE files
** Author(s): Jeremy Magland
** Created: 4/1/2016
*******************************************************/


#ifndef BLOCKNUFFT3D_C_H
#define BLOCKNUFFT3D_C_H

extern "C" {

    //the Fortran interface
    //plan is the output, which is a pointer to a class
    //on the fortran size it should be considered a pointer to a 64-bit integer
    void blocknufft3d_create_plan_f_(
            void **plan,
            int *N1,int *N2,int *N3,int *M,
            double *x,double *y,double *z,
            double *eps);
    void blocknufft3d_run_f_(void **plan,double *uniform_d,double *nonuniform_d);
    void blocknufft3d_destroy_plan_f_(void **plan);
}

#endif // BLOCKNUFFT3D_C_H

