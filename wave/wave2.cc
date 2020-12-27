#include <petsc.h>
#include <cmath>

struct Field
{
  PetscReal u, v;
};

struct WaveCtx
{
  PetscReal C;            //wave speed
};

//initial perturbation:
struct Perturbation
{
  PetscReal operator()(PetscReal x, PetscReal y)
  {
    return std::exp(-std::log(2.0) * ((x - xs) * (x - xs) + (y - ys) * (y - ys)) / (w * w));
  };

  PetscReal xs, ys; //source location:
  PetscReal w;      //half width:
};

PetscErrorCode Spacings(DMDALocalInfo *info, PetscReal *hx, PetscReal *hy);
PetscErrorCode FormRHSFunctionLocal(DMDALocalInfo *info, PetscReal t, Field **aX, Field **aF, WaveCtx *user);
template <typename Function>
PetscErrorCode initial_condition(Function f, DM da, Vec X);
extern PetscErrorCode MyTSMonitor(TS,PetscInt,PetscReal,Vec,void*);


int main(int argc,char **argv)
{
    PetscErrorCode        ierr;
    TS                    ts;
    Vec                   x;
    DM                    da;
    DMDALocalInfo         info;

    //grid:
    PetscReal             Lx = 10.0, Ly = 10.0;
    PetscInt              Nx = 640, Ny = 640;
    PetscReal             hx = Lx/PetscReal(Nx), hy = Ly/PetscReal(Ny);

    //wave speed, time step and cfl:
    PetscReal             C = 0.5, dt = 0.001, Tf = 1.0;
    PetscReal             cfl = C*(dt/hx + dt/hy);  

    ierr = PetscInitialize(&argc,&argv,NULL,NULL); if (ierr) return ierr;

    //set domain size, ngpts and wave speed:
    WaveCtx               user{C};

    //setup distributed grid:
    ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC,  DMDA_STENCIL_STAR, Nx, Ny, PETSC_DECIDE, PETSC_DECIDE, 2, 1, NULL, NULL, &da);  
    ierr = DMSetFromOptions(da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);
    ierr = DMDASetFieldName(da,0,"u"); CHKERRQ(ierr);
    ierr = DMDASetFieldName(da,1,"v"); CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
    ierr = DMDASetUniformCoordinates(da, 0.0, Lx, 0.0, Ly, -1.0, -1.0); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "running on %d x %d grid with hx = %.6f and hy = %.6f...\n", info.mx,info.my,hx,hy); CHKERRQ(ierr);

    //setup ts:
    ierr = TSCreate(PETSC_COMM_WORLD,&ts); CHKERRQ(ierr);
    ierr = TSSetProblemType(ts,TS_NONLINEAR); CHKERRQ(ierr);
    ierr = TSSetDM(ts,da); CHKERRQ(ierr);
    ierr = TSSetApplicationContext(ts,&user); CHKERRQ(ierr);
    ierr = DMDATSSetRHSFunctionLocal(da,INSERT_VALUES,(DMDATSRHSFunctionLocal)FormRHSFunctionLocal,&user); CHKERRQ(ierr);
    ierr = TSMonitorSet(ts,MyTSMonitor,0,0);CHKERRQ(ierr);

    ierr = TSSetType(ts,TSRK);CHKERRQ(ierr);
    ierr = TSSetTime(ts,0.0); CHKERRQ(ierr);
    ierr = TSSetMaxTime(ts,Tf); CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,dt); CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP); CHKERRQ(ierr);
    ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

    //cfl number:
    ierr = PetscPrintf(PETSC_COMM_WORLD, "cfl = %.6f \n", cfl); CHKERRQ(ierr);

    //initial condition:
    ierr = DMCreateGlobalVector(da,&x); CHKERRQ(ierr);
    ierr = initial_condition(Perturbation{0.5*Lx,0.5*Ly,0.125},da,x);

    //solve:
    ierr = TSSolve(ts,x); CHKERRQ(ierr);


    //destroy petsc objects:
     VecDestroy(&x);  TSDestroy(&ts);  
     DMDestroy(&da);
    return PetscFinalize();
};

PetscErrorCode Spacings(DMDALocalInfo *info, PetscReal *hx, PetscReal *hy) {
    *hx = 1.0 / (PetscReal)(info->mx);
    *hy = 1.0 / (PetscReal)(info->my);
    return 0;
}

//form right hand side of the wave equation: dX/dt = F(X)
PetscErrorCode FormRHSFunctionLocal(DMDALocalInfo *info, PetscReal t, Field **aX, Field **aF, WaveCtx *user)
{
  //wave speed:
  PetscReal C = user->C;

  //grid size:
  PetscReal hx, hy;
  Spacings(info, &hx, &hy);

  for (int j = info->ys; j < info->ys + info->ym; j++)
  {
    for (int i = info->xs; i < info->xs + info->xm; i++)
    {

      auto uxx = (aX[j][i + 1].u - 2.0 * aX[j][i].u + aX[j][i - 1].u) / (hx * hx);
      auto uyy = (aX[j + 1][i].u - 2.0 * aX[j][i].u + aX[j - 1][i].u) / (hy * hy);

      aF[j][i].u = aX[j][i].v;
      aF[j][i].v = C * C * (uxx + uyy);
    }
  }
  return 0;
}

//initial condition:
template <typename Function>
PetscErrorCode initial_condition(Function f, DM da, Vec X)
{
  PetscErrorCode    ierr;
  DMDALocalInfo     info;
  DMDACoor2d        **aC;  
  Field             **aX;  

  //get grid info:
  ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);

  //get grid coordinates:
  ierr = DMDAGetCoordinateArray(da,&aC); CHKERRQ(ierr);

  //get array from vector:
  ierr = DMDAVecGetArray(da,X,&aX);  CHKERRQ(ierr);

  //loop over current process:
  for (int j = info.ys; j < info.ys+info.ym; j++) {
    for (int i = info.xs; i < info.xs+info.xm; i++) {
      aX[j][i].u = f(aC[j][i].x,aC[j][i].y);
    }
  }

  ierr = DMDAVecRestoreArray(da,X,&aX); CHKERRQ(ierr);
  ierr = DMDARestoreCoordinateArray(da,&aC); CHKERRQ(ierr);
  return 0;
}

PetscErrorCode MyTSMonitor(TS ts,PetscInt step,PetscReal ptime,Vec v,void *ctx)
{
  PetscErrorCode ierr;
  DM da;
  ierr = TSGetDM(ts,&da); CHKERRQ(ierr);
  if (step % 10 == 0)
  {
    //plot vtk:
    char filename[20];
    sprintf(filename, "sol-%05d.vtk", step); // 4 is the padding level, increase it for longer simulations 
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Writing data in vtk format to %s at t = %f, step = %d\n", filename, ptime, 1); CHKERRQ(ierr);
    PetscViewer viewer;  
    PetscViewerASCIIOpen(PETSC_COMM_WORLD, filename, &viewer);
    PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_VTK);
    ierr = DMView(da, viewer);
    VecView(v, viewer);  
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  }

  return 0;
}