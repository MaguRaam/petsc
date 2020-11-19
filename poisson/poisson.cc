/**
 * @file poisson.cc
 * @author Magu
 * @brief Solve Poisson equation on a distributed structured grid:
 *          −∇2u = f  on S,
 *        Bcs:
 *          u = 0   on ∂S.                  
 * @version 0.1
 * @date 2020-11-19
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#include <petsc.h>
#include <iostream>

/**
 * @brief To compile and run
 * mkdir build
 * cd build 
 * cmake ..
 * make
 * 
 * mpiexec -n P ./poisson -da_grid_x MX -da_grid_y MY
 * or
 * mpiexec -n P ./poisson -da_refine 2
 */

/**
 * @brief Form coeffiecient Matrix A 
 * 
 * @param da 
 * @param A 
 * @return PetscErrorCode 
 */
PetscErrorCode formMatrix(DM da, Mat A)
{
    /**
     * @brief The info object stores the local and global range of the grid points
     * 
     * The local process owns info.xm × info.ym rectangular subgrid with range
     *  info.xs ≤ i ≤ info.xs + info.xm − 1,
     *  info.ys ≤ j ≤ info.ys + info.ym − 1
     * 
     * The global range:
     * 0 ≤ i ≤ info.mx - 1
     * 0 ≤ j ≤ info.my - 1
     * 
     * Example loop over the sub grid owned by current process:
     * for (j=info.ys; j<info.ys+info.ym; j++) {
     *  for (i=info.xs; i<info.xs+info.xm; i++){
     *      Do something at grid point (i,j)
     *  } 
     * }
     */
    DMDALocalInfo info;
    DMDAGetLocalInfo(da, &info);

    //grid spacing hx and hy
    double hx = 1.0 / (info.mx - 1), hy = 1.0 / (info.my - 1);

    /**
     * @brief MatStencil is a struct that stores grid index (i,j)
     * 
     * We store the matrix row index that is the center grid point (i,j)
     * And 5 column indices those are 
     * (i,j) (i-1,j) (i+1,j) (i,j-1) (i,j+1) 
     * 
     * Note: we don't worry about the actual row and col index of the matrix A 
     * we use only the grid index, Petsc will do the conversion.
     */

    MatStencil row, col[5];

    //coefficient array
    double v[5];

    //coefficients are written using a and b:
    double ac = 2 * ((hy / hx) + (hx / hy)), aw = -hy / hx, ae = -hy / hx, an = -hx / hy, as = -hx / hy;

    int ncols = 1;
    //loop over grid points in the current process:
    for (int j = info.ys; j < info.ys + info.ym; j++)
    {
        for (int i = info.xs; i < info.xs + info.xm; i++)
        {
            //get row index (i,j)
            row.j = j;
            row.i = i;

            //set the diagonal coefficient:

            //get col index for the diagonal entry (i,j)
            col[0].i = i;
            col[0].j = j;

            //check whether (i,j) is on the boundary:
            //if it is on the boundary the diagonal element is 1:
            ncols = 1;
            if (i == 0 || i == info.mx - 1 || j == 0 || j == info.my - 1)
                v[0] = 1.0;
            else
            {
                v[0] = ac;

                //fill i-1,j coefficient:
                if (i - 1 > 0)
                {
                    //get col index:
                    col[ncols].i = i - 1;
                    col[ncols].j = j;
                    v[ncols++] = aw;
                }

                //fill i+1,j coefficient:
                if (i + 1 < info.mx - 1)
                {
                    //get col index:
                    col[ncols].i = i + 1;
                    col[ncols].j = j;
                    v[ncols++] = ae;
                }

                //fill i,j-1 coefficients:
                if (j - 1 > 0)
                {
                    col[ncols].i = i;
                    col[ncols].j = j - 1;
                    v[ncols++] = as;
                }

                //fill i,j+1 coefficients:
                if (j + 1 < info.my - 1)
                {
                    //get col index:
                    col[ncols].i = i;
                    col[ncols].j = j + 1;
                    v[ncols++] = an;
                }
            }

            MatSetValuesStencil(A, 1, &row, ncols, col, v, INSERT_VALUES);
        }
    }

    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    return 0;
}

PetscErrorCode formExact(DM da, Vec uexact)
{
    /**
     * @brief The info object stores the local and global range of the grid points
     * 
     * The local process owns info.xm × info.ym rectangular subgrid with range
     *  info.xs ≤ i ≤ info.xs + info.xm − 1,
     *  info.ys ≤ j ≤ info.ys + info.ym − 1
     * 
     * The global range:
     * 0 ≤ i ≤ info.mx - 1
     * 0 ≤ j ≤ info.my - 1
     * 
     * Example loop over the sub grid owned by current process:
     * for (j=info.ys; j<info.ys+info.ym; j++) {
     *  for (i=info.xs; i<info.xs+info.xm; i++){
     *      Do something at grid point (i,j)
     *  } 
     * }
     */
    DMDALocalInfo info;
    DMDAGetLocalInfo(da, &info);

    //grid spacing hx and hy
    double hx = 1.0 / (info.mx - 1), hy = 1.0 / (info.my - 1);

    //get 2D array from vector:
    double **auexact;

    double x, y;

    DMDAVecGetArray(da, uexact, &auexact);
    for (int j = info.ys; j < info.ys + info.ym; j++)
    {
        y = j * hy;
        for (int i = info.xs; i < info.xs + info.xm; i++)
        {
            x = i * hx;
            auexact[j][i] = x * x * (1.0 - x * x) * y * y * (y * y - 1.0);
        }
    }
    DMDAVecRestoreArray(da, uexact, &auexact);
    return 0;
}

PetscErrorCode formRHS(DM da, Vec b)
{
    DMDALocalInfo info;
    DMDAGetLocalInfo(da, &info);

    //grid spacing hx and hy
    double hx = 1.0 / (info.mx - 1), hy = 1.0 / (info.my - 1);

    //get 2D array from vector:
    double **ab;

    double f, x, y;

    DMDAVecGetArray(da, b, &ab);
    for (int j = info.ys; j < info.ys + info.ym; j++)
    {
        y = j * hy;
        for (int i = info.xs; i < info.xs + info.xm; i++)
        {
            x = i * hx;
            if (i == 0 || i == info.mx - 1 || j == 0 || j == info.my - 1)
            {
                ab[j][i] = 0.0; // on boundary: 1*u = 0
            }
            else
            {
                f = 2.0 * ((1.0 - 6.0 * x * x) * y * y * (1.0 - y * y) + (1.0 - 6.0 * y * y) * x * x * (1.0 - x * x));
                ab[j][i] = hx * hy * f;
            }
        }
    }
    DMDAVecRestoreArray(da, b, &ab);
    return 0;
}

int main(int argc, char **args)
{
    PetscErrorCode ierr;
    int rank;

    DM da;
    Mat A;
    Vec b, u, uexact;
    KSP ksp;

    ierr = PetscInitialize(&argc, &args, NULL, NULL);
    if (ierr) return ierr;

    MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

    //generate distributed structured grid
    //change default 9x9 size using -da_grid_x M -da_grid_y N
    ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, 9, 9, PETSC_DECIDE,
                        PETSC_DECIDE, 1, 1, NULL, NULL, &da);
    CHKERRQ(ierr);
    DMSetFromOptions(da);
    DMSetUp(da);

    //create A matrix using DM object:
    DMCreateMatrix(da, &A);
    MatSetFromOptions(A);

    //create b, u, uexact using DM object:
    DMCreateGlobalVector(da, &b);
    VecDuplicate(b, &u);
    VecDuplicate(b, &uexact);

    //fill matrix A and vector u, uexact and b:
    formMatrix(da, A); //use MatView(A,PETSC_VIEWER_STDOUT_WORLD) to print A;
    formRHS(da,b);
    formExact(da,uexact);

    //solve Ax = b;
    KSPCreate(PETSC_COMM_WORLD,&ksp);
    KSPSetOperators(ksp,A,A);
    KSPSetFromOptions(ksp);
    KSPSolve(ksp,b,u);

    //error norm ||u-uexact||
    double errnorm;
    VecAXPY(u,-1.0,uexact);
    VecNorm(u,NORM_INFINITY,&errnorm);

    DMDALocalInfo info;
    DMDAGetLocalInfo(da,&info);

    if (rank == 0) std::cout<<"on "<<info.mx<<"x"<<info.my<<"\t|u - uexact|_inf = "<<errnorm<<"\n";    


    //destroy objects:
    VecDestroy(&u);
    VecDestroy(&uexact);
    VecDestroy(&b);
    MatDestroy(&A);
    DMDestroy(&da);

    return PetscFinalize();
}
