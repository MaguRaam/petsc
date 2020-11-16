//Petsc:
#include <petsc.h>

//c++
#include <iostream>
#include <array>
#include <algorithm>
#include <numeric>

template<typename Vector>
void print(const Vector& v){
    for (const auto& e : v) std::cout<<e<<" ";
    std::cout<<"\n";
}


int main(int argc, char *argv[])
{
    Vec x, b;
    const int size = 10;
    int low,high;

    //set index:
    std::array<int,size> index;
    std::iota(index.begin(),index.end(),0);

    //values:
    std::array<double,size> values;     

    PetscInitialize(&argc, &argv, NULL, NULL);
    
    //Create x vector:
    VecCreate(PETSC_COMM_WORLD,&x);
    VecSetSizes(x,PETSC_DECIDE,size);
    VecSetFromOptions(x);
    VecSet(x,10.0);     
    VecAssemblyBegin(x);
    VecAssemblyEnd(x);
    VecView(x,PETSC_VIEWER_STDOUT_WORLD);

    //Create b vector:
    VecDuplicate(x,&b);

    //access local array in each process:
    int local_size;
    VecGetLocalSize(x,&local_size); 
    std::cout<<"local size = "<<local_size<<std::endl;

    VecDestroy(&x);
    VecDestroy(&b);
    return PetscFinalize();
    return 0;
}
