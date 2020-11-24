/**
 * @file ode.cpp
 * @author Magu (you@domain.com)
 * @brief Solve
 * y = Ay + f (t),
 * 
 * A = [0 1; -1 0]
 * f(t) = [0;t]
 * 
 * 
 * @version 0.1
 * @date 2020-11-19
 * 
 * @copyright Copyright (c) 2020
 * 
 */
#include <petsc.h>
#include <iostream>

/**
 * @brief The right hand side function of the ODE that is Ay + f (t)
 * 
 * @param ts petsc TS objec
 * @param t time
 * @param y solution vector at time t
 * @param g evaluted right hand side
 * @param ptr 
 * @return PetscErrorCode 
 */
PetscErrorCode FormRHSFunction(TS , double t, Vec y, Vec g, void* )
{
	const double* ay;
	double* ag;

	VecGetArrayRead(y,&ay);
	VecGetArray(g,&ag);

	ag[0] = ay[1];
	ag[1] = -ay[0] + t;
	VecRestoreArrayRead(y,&ay);
	VecRestoreArray(g,&ag);
	return 0;
}

/**
 * @brief Exact solution at time t
 * 
 * @param t time 
 * @param y exact solution vector at t
 * @return PetscErrorCode 
 */
PetscErrorCode ExactSolution(double t, Vec y)
{
	double* ay;
	VecGetArray(y,&ay);
	ay[0] = t - sin(t);
	ay[1] = 1.0 - cos(t);
	VecRestoreArray(y,&ay);
	return 0;
}


int main(int argc,char **argv) 
{
	PetscErrorCode ierr;
	int rank;
	Vec y, yexact;

	TS ts;

	ierr = PetscInitialize(&argc,&argv,NULL,NULL); if (ierr) return ierr;
	MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

	//Create y and yexact vector and set size = 2
	VecCreate(PETSC_COMM_WORLD,&y);
	VecSetSizes(y,PETSC_DECIDE,2);
	VecSetFromOptions(y);
	VecDuplicate(y,&yexact);

	//Construct TS object
	TSCreate(PETSC_COMM_WORLD,&ts);
	TSSetProblemType(ts,TS_NONLINEAR);
	TSSetRHSFunction(ts,NULL,FormRHSFunction,NULL);
	TSSetType(ts,TSRK); //Rungekutta family solvers

	//Note: this default choice can be overridden by run-time option 
	//-ts type and the particular solver can be set by -ts_rk_type

	//set time axis:
	double to = 0.0; //initial time
	TSSetTime(ts,to);
	double tf = 20.0; //final time
	TSSetMaxTime(ts,tf);
	double dt = 0.1;
	TSSetTimeStep(ts,dt);
	TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);
	TSSetFromOptions(ts);

	//set initial condition:
	TSGetTime(ts,&to);
	ExactSolution(to,y);

	//solve ode:
	TSSolve(ts,y);

	//compute error and print:
	int steps;
	TSGetStepNumber(ts,&steps);
	TSGetTime(ts,&tf); //get current time
	ExactSolution(tf,yexact);
	VecAXPY(y,-1.0,yexact);   //y = y - yexact
	double err;
	VecNorm(y,NORM_INFINITY,&err);

	if (rank == 0) std::cout<< "error at tf = "<<tf<<" with "<<steps<<"steps "<<" |y - yexact|_inf "<<err<<std::endl;



	//Destroy all objects:
	VecDestroy(&y);
	VecDestroy(&yexact);
	TSDestroy(&ts);

	return PetscFinalize();
}
