/******************************************************
** See the accompanying README and LICENSE files
** Author(s): Jeremy Magland
** Created: 4/1/2016
*******************************************************/

#include "blocknufft3d_c.h"
#include "blocknufft3d.h"
#include <stdio.h>

void blocknufft3d_create_plan_f_(void **plan, int *N1, int *N2, int *N3, int *M, double *x, double *y, double *z, double *eps)
{
    printf("%s: %dx%dx%d\n",__FUNCTION__,*N1,*N2,*N3);
    *plan=blocknufft3d_create_plan(*N1,*N2,*N3,*M,x,y,z,*eps);
}

void blocknufft3d_run_f_(void **plan, double *uniform_d, double *nonuniform_d)
{
    printf("%s\n",__FUNCTION__);
    blocknufft3d_run(*plan,uniform_d,nonuniform_d);
}

void blocknufft3d_destroy_plan_f_(void **plan)
{
    printf("%s\n",__FUNCTION__);
    blocknufft3d_destroy_plan(*plan);
}
