/**
 * @file expcircle.cc
 * @author Magu (you@domain.com)
 * @brief Solve Nonlinear equation using Newton's method with user written Jacobian:
 * @version 0.1
 * @date 2020-11-20
 * 
 * @copyright Copyright (c) 2020
 * 
 */
#include <petsc.h>

//parameter in struct:
struct AppCtx{double b;};




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
    AppCtx* user = (AppCtx*)ctx;
    const double b = user->b;

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

PetscErrorCode FormJacobian(SNES snes, Vec x, Mat J, Mat P, void* ctx)
{
   PetscErrorCode ierr;
    AppCtx           *user = (AppCtx*)ctx;
    const PetscReal  b = user->b, *ax;
    PetscReal        v[4];
    PetscInt         row[2] = {0,1}, col[2] = {0,1};

    ierr = VecGetArrayRead(x,&ax); CHKERRQ(ierr);
    v[0] = PetscExpReal(b * ax[0]);  v[1] = -1.0;
    v[2] = 2.0 * ax[0];              v[3] = 2.0 * ax[1];
    ierr = VecRestoreArrayRead(x,&ax); CHKERRQ(ierr);
    ierr = MatSetValues(P,2,row,2,col,v,INSERT_VALUES); CHKERRQ(ierr);
    ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    if (J != P) {
        ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    }
return 0;
}



int main(int argc, char** argv)
{
    PetscErrorCode ierr;
  SNES   snes;         // nonlinear solver
  Vec    x,r;          // solution, residual vectors
  Mat    J;
  AppCtx user;

  ierr = PetscInitialize(&argc,&argv,NULL,NULL); if (ierr) return ierr;
  user.b = 2.0;

  ierr = VecCreate(PETSC_COMM_WORLD,&x); CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,2); CHKERRQ(ierr);
  ierr = VecSetFromOptions(x); CHKERRQ(ierr);
  ierr = VecDuplicate(x,&r); CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&J); CHKERRQ(ierr);
  ierr = MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,2,2); CHKERRQ(ierr);
  ierr = MatSetFromOptions(J); CHKERRQ(ierr);
  ierr = MatSetUp(J); CHKERRQ(ierr);

  ierr = SNESCreate(PETSC_COMM_WORLD,&snes); CHKERRQ(ierr);
  ierr = SNESSetFunction(snes,r,FormFunction,&user);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,J,J,FormJacobian,&user);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  ierr = VecSet(x,1.0);CHKERRQ(ierr);            // initial iterate
  ierr = SNESSolve(snes,NULL,x);CHKERRQ(ierr);
  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

  SNESDestroy(&snes);  MatDestroy(&J);  VecDestroy(&x);  VecDestroy(&r);
return PetscFinalize();
}

/**
 * @brief To run the code
 * ./expcircle -snes_fd -snes_monitor
 * 
 */