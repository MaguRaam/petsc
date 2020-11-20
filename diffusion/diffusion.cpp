/**
 * @file diffusion.cpp
 * @author Magu (you@domain.com)
 * @brief Solves diffusion equation in 2d
 * 
 * ∂u/∂t = k∇2u + f(x,y) 
 * 
 * where, f(x,y) = 3e^(−25(x−0.6))sin(2πy).
 * 
 * ics:
 *     u(0,x,y) = 0 
 * 
 * bcs:
 *  ∂u/∂x = sin(6πy) x = 0
 *  ∂u/∂x = 0
 *  top and bottom are periodic:
 * 
 * @version 0.1
 * @date 2020-11-19
 * 
 * @copyright Copyright (c) 2020
 * 
 */

PetscReal f_source(PetscReal x, PetscReal y)
{
    return 3.0 * PetscExpReal(-25.0 * (x - 0.6) * (x - 0.6)) * PetscSinReal(2.0 * PETSC_PI * y);
}

PetscReal gamma_neumann(PetscReal y)
{
    return PetscSinReal(6.0 * PETSC_PI * y);
}


#include <petsc.h>

int main(int argc, char **argv)
{
    PetscErrorCode ierr;
    TS ts;
    Vec u;
    DM da;
    
    ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr)return ierr;

    //start DM setup
    ierr = DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_PERIODIC,DMDA_STENCIL_STAR,5,4,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da);CHKERRQ(ierr);
    ierr = DMSetFromOptions(da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(da,&u); CHKERRQ(ierr);
    //end DM setup

    //start TS setup
    ierr = TSCreate(PETSC_COMM_WORLD,&ts); CHKERRQ(ierr);
    ierr = TSSetProblemType(ts,TS_NONLINEAR); CHKERRQ(ierr);
    ierr = TSSetDM(ts,da); CHKERRQ(ierr);
    ierr = DMDATSSetRHSFunctionLocal(da,INSERT_VALUES,(DMDATSRHSFunctionLocal)FormRHSFunctionLocal,&user); CHKERRQ(ierr); 








    return PetscFinalize();
}