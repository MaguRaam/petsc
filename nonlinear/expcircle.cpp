/**
 * @file expcircle.cc
 * @author Magu (you@domain.com)
 * @brief Solve Nonlinear equation using Newton's method:
 * @version 0.1
 * @date 2020-11-20
 * 
 * @copyright Copyright (c) 2020
 * 
 */
#include <petsc.h>



/**
 * @brief Newton's Method:
 * 
 * Solve F(x) = 0
 *
 * at each iterate solve a Linear system for s:
 * ~ Jf(xk)*s = −F(xk)
 * 
 * And update the soution xk -> xk+1
 * ~ xk+1 = xk + λks
 *
 * problem:
 *  F(x) = [1/b e^(bxo)- x1, xo*xo + x1*x1 - 1] 
 *  Jf(x) = [e^(bxo) -1, 2xo 2x1]
 *  
 */


/**
 * @brief FormFunction() takes x as the first Vec argument and it generates
output F(x) as the second Vec
 * 
 * @param snes Nonlinear solver object
 * @param x    current state
 * @param F    residual at current state F(x) 
 * @param ctx 
 * @return PetscErrorCode 
 */
PetscErrorCode FormFunction(SNES snes, Vec x, Vec F, void *ctx)
{
    const double b = 2.0;
    const double* ax;
    double* aF;

    VecGetArrayRead(x,&ax);
    VecGetArray(F,&aF);
    aF[0] = (1.0/b)*PetscExpReal(b*ax[0]) - ax[1];
    aF[1] = ax[0]*ax[0] + ax[1]*ax[1] - 1.0;
    VecRestoreArrayRead(x,&ax);
    VecRestoreArray(F,&aF);
    return 0;    
}




int main(int argc, char** argv)
{
    SNES snes;
    Vec x,r;   //solution and residual vector:
    PetscInitialize(&argc, &argv, NULL, NULL);

    //setup solution x and residual r
    VecCreate(PETSC_COMM_WORLD,&x);
    VecSetSizes(x,PETSC_DECIDE,2);
    VecSetFromOptions(x);
    VecSet(x,1.0);
    VecDuplicate(x,&r);

    //setup snes and solve f(x) = 0:
    SNESCreate(PETSC_COMM_WORLD, &snes);
    SNESSetFunction(snes, r, FormFunction, NULL);
    SNESSetFromOptions(snes);
    SNESSolve(snes,NULL,x);
    VecView(x,PETSC_VIEWER_STDOUT_WORLD);

    //Destroy objects:
    VecDestroy(&x);
    VecDestroy(&r);
    SNESDestroy(&snes);

    return PetscFinalize();
}

/**
 * @brief To run the code
 * ./expcircle -snes_fd -snes_monitor
 * 
 */