/**
 * @file diffusion.cpp
 * @author Solves the diffusion equation
 * @brief 
 * @version 0.1
 * @date 2020-11-21
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#include <petsc.h>

struct HeatCtx{PetscReal D0;};

PetscErrorCode FormRHSFunctionLocal(DMDALocalInfo *, PetscReal, PetscReal **, PetscReal **, HeatCtx *);

PetscReal f_source(PetscReal x, PetscReal y)
{
    return 3.0 * PetscExpReal(-25.0 * (x - 0.6) * (x - 0.6)) * PetscSinReal(2.0 * PETSC_PI * y);
}

int main(int argc, char **argv)
{
    PetscErrorCode ierr;
    HeatCtx user;
    TS ts;
    Vec u;
    DM da;
    DMDALocalInfo info;
    PetscReal t0, tf;

    ierr = PetscInitialize(&argc, &argv, NULL, NULL);
    if (ierr)
        return ierr;

    //set diffusion constant:
    user.D0 = 1.0;

    //create DMDA grid:
    ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, 10, 10, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da);CHKERRQ(ierr);
    ierr = DMSetFromOptions(da);CHKERRQ(ierr);
    ierr = DMSetUp(da);CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(da, &u);CHKERRQ(ierr);
    const double L = 1.0;
    ierr = DMDASetUniformCoordinates(da, 0.0, L, 0.0, L, -1.0, -1.0); CHKERRQ(ierr);


    //create TS object
    ierr = TSCreate(PETSC_COMM_WORLD, &ts);CHKERRQ(ierr);
    ierr = TSSetProblemType(ts, TS_NONLINEAR);CHKERRQ(ierr);
    ierr = TSSetDM(ts, da);CHKERRQ(ierr);
    ierr = TSSetApplicationContext(ts, &user);CHKERRQ(ierr);
    ierr = DMDATSSetRHSFunctionLocal(da, INSERT_VALUES, (DMDATSRHSFunctionLocal)FormRHSFunctionLocal, &user);CHKERRQ(ierr);
    ierr = TSSetType(ts, TSEULER);CHKERRQ(ierr);
    ierr = TSSetTime(ts, 0.0);CHKERRQ(ierr);
    ierr = TSSetMaxTime(ts, 0.5);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts, 0.001);CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
    ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

    // report on set up
    ierr = TSGetTime(ts, &t0);CHKERRQ(ierr);CHKERRQ(ierr);
    ierr = TSGetMaxTime(ts, &tf);
    ierr = DMDAGetLocalInfo(da, &info);CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD, "solving on %d x %d grid for t0=%g to tf=%g ...\n", info.mx, info.my, t0, tf);CHKERRQ(ierr);

    //solve
    ierr = VecSet(u, 0.0);CHKERRQ(ierr);
    ierr = TSSolve(ts, u);CHKERRQ(ierr);
	
	 //plot vtk:
    char filename[20];
    sprintf(filename, "sol-%05d.vtk", 1); // 4 is the padding level, increase it for longer simulations 
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Writing data in vtk format to %s at t = %f, step = %d\n", filename, time, 1); CHKERRQ(ierr);
    PetscViewer viewer;  
    PetscViewerASCIIOpen(PETSC_COMM_WORLD, filename, &viewer);
    PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_VTK);
    ierr = DMView(da, viewer);
    VecView(u, viewer);
    
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);


    VecDestroy(&u);
    TSDestroy(&ts);
    DMDestroy(&da);
    return PetscFinalize();
}



PetscErrorCode FormRHSFunctionLocal(DMDALocalInfo *info, PetscReal t, PetscReal **au, PetscReal **aG, HeatCtx *user)
{
    PetscErrorCode ierr;

    PetscReal hx = 1.0 / (PetscReal)(info->mx - 1);
    PetscReal hy = 1.0 / (PetscReal)(info->my - 1);

    PetscReal x, y;

    PetscReal uxx, uyy;

    //loop over locally owned grid points:
    for (PetscInt j = info->ys; j < info->ys + info->ym; j++)
    {
        y = hy * j;
        for (PetscInt i = info->xs; i < info->xs + info->xm; i++)
        {
            x = hx * i;

            //check for boundary nodes
            if (i == 0 || j == 0 || i == info->mx - 1 || j == info->my - 1)
                aG[j][i] = au[j][i];
            else
            {
                uxx = (au[j][i + 1] - 2.0 * au[j][i] + au[j][i - 1]) / (hx * hx);
                uyy = (au[j + 1][i] - 2.0 * au[j][i] + au[j - 1][i]) / (hx * hx);

                aG[j][i] = user->D0 * (uxx + uyy) + f_source(x, y);
            }
        }
    }
    return 0;
}

