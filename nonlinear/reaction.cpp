/**
 * @file reaction.cpp
 * @author Magu 
 * @brief Solves 1d steady state 1d reaction diffusion equation:
 *         -u" + ρ sqrt(u) = 0
 * @version 0.1
 * @date 2020-11-20
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#include <petsc.h>

/**
 * @brief Discretization:
 * -(uj+1 − 2uj + uj−1)/(h*h) - R(uj) = f(xj)
 */

struct AppCtx
{
    PetscReal rho, M, alpha, beta;
    PetscBool noRinJ;
};

PetscReal f_source(PetscReal);
PetscErrorCode InitialAndExact(DMDALocalInfo *, PetscReal *, PetscReal *, AppCtx *);
PetscErrorCode FormFunctionLocal(DMDALocalInfo *, PetscReal *, PetscReal *, AppCtx *);

int main(int argc, char **args)
{

    DM da;
    SNES snes;
    AppCtx user;
    Vec u, uexact;
    PetscReal errnorm, *au, *auex;
    DMDALocalInfo info;

    PetscInitialize(&argc, &args, NULL, NULL);

    //set parameters:
    user.rho = 10.0;
    user.M = PetscSqr(user.rho / 12.0);
    user.alpha = user.M;
    user.beta = 16.0 * user.M;
    user.noRinJ = PETSC_FALSE;

    //create 1d grid:
    DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, 9, 1, 1, NULL, &da);
    DMSetFromOptions(da);
    DMSetUp(da);
    DMSetApplicationContext(da, &user);

    //create vector from grid and get the array:
    DMCreateGlobalVector(da, &u);
    VecDuplicate(u, &uexact);

    //set info object:
    DMDAGetLocalInfo(da, &info);

    DMDAVecGetArray(da, u, &au);
    DMDAVecGetArray(da, uexact, &auex);
    InitialAndExact(&info, au, auex, &user);
    DMDAVecRestoreArray(da, u, &au);
    DMDAVecRestoreArray(da, uexact, &auex);

    //create nonlinear solver object:
    SNESCreate(PETSC_COMM_WORLD, &snes);
    SNESSetDM(snes, da);

    //destroy petsc objects:
    VecDestroy(&u);
    VecDestroy(&uexact);
    SNESDestroy(&snes);
    DMDestroy(&da);

    return PetscFinalize();
}

PetscReal f_source(PetscReal x) { return 0.0; }

PetscErrorCode InitialAndExact(DMDALocalInfo *info, PetscReal *u0, PetscReal *uex, AppCtx *user)
{
    PetscInt i;
    PetscReal h = 1.0 / (info->mx - 1), x;

    //loop over local grid points:
    for (i = info->xs; i < info->xs + info->xm; i++)
    {
        x = i * h;
        u0[i] = user->alpha * (1.0 - x) + user->beta * x;
        uex[i] = user->M * PetscPowReal(x + 1.0, 4.0);
    }

    return 0;
}

PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, PetscReal *u, PetscReal *FF, AppCtx *user)
{
    PetscInt i;
    PetscReal h = 1.0 / (info->mx - 1), x, R;

    //loop over local grid points:
    for (i = info->xs; i < info->xs + info->xm; i++)
    {
        if (i == 0)
            FF[i] = u[i] - user->alpha;
        else if (i == info->mx - 1)
            FF[i] = u[i] - user->beta;
        else
        {
            if (i == 1)
                FF[i] = -u[i + 1] + 2.0 * u[i] - user->alpha;
            else if (i == info->mx - 2)
                FF[i] = -user->beta + 2.0 * u[i] - u[i - 1];
            else
                FF[i] = -u[i + 1] + 2.0 * u[i] - u[i - 1];
        }
        R = -user->rho * PetscSqrtReal(u[i]);
        x = i * h;
        FF[i] -= h * h * (R + f_source(x));
    }
}