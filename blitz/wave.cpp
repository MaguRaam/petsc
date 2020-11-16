/*****************************************************************************
 * wave *
 *****************************************************************************/

#include "blitz/array.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>

using namespace blitz;

std::string int_to_string(unsigned int value, const unsigned int digits)
{
    std::string lc_string = std::to_string(value);

    if (lc_string.size() < digits)
    {
        // We have to add the padding zeroes in front of the number
        const unsigned int padding_position = (lc_string[0] == '-')
                                                  ? 1
                                                  : 0;

        const std::string padding(digits - lc_string.size(), '0');
        lc_string.insert(padding_position, padding);
    }

    return lc_string;
}

int main()
{
    //domain:
    const int dim = 2;
    double Lx = 1.0, Ly = 1.0;
    int Nx = 100, Ny = 100;
    double hx = Lx / Nx, hy = Ly / Ny;

    Array<double, 1> x(Nx), y(Ny);

    for (int i = 0; i < Nx; ++i)
        x(i) = i * hx;
    for (int j = 0; j < Ny; ++j)
        y(j) = j * hy;

    //time:
    double T = 1;
    double dt = 0.001;
    double t = 0.0;
    int n = 0;

    //initialize u:
    double k = 4, l = 4;
    auto uf = [k, l](double x, double y) { return sin(M_PI * k * x) * sin(M_PI * l * y); };

    Array<double, dim> u(Nx, Ny), v(Nx, Ny), du_dt(Nx, Ny), dv_dt(Nx, Ny);

    for (int i = 0; i < Nx; ++i)
    {
        for (int j = 0; j < Ny; ++j)
        {
            u(i, j) = uf(x(i), y(j));
            v(i, j) = 0.0;
            du_dt(i, j) = 0.0;
            dv_dt(i, j) = 0.0;
        }
    }

    while (t < T)
    {
        std::cout << "time = " << t << std::endl;

        //loop over interior grid points and compute time derivative and update:
        for (int i = 1; i < Nx - 1; ++i)
        {
            for (int j = 1; j < Ny - 1; ++j)
            {
                du_dt(i, j) = v(i, j);
                dv_dt(i, j) = (u(i + 1, j) + u(i - 1, j) + u(i, j + 1) + u(i, j - 1) - 4.0 * u(i, j)) / (hx * hy);
            }
        }

        //loop over interior grid points and update the solution:
        for (int i = 1; i < Nx - 1; ++i)
        {
            for (int j = 1; j < Ny - 1; ++j)
            {

                u(i, j) = u(i, j) + dt * (du_dt(i, j));
                v(i, j) = v(i, j) + dt * (dv_dt(i, j));
            }
        }

        //plot:
        if (n % 100 == 0)
        {
            std::ofstream tpl;
            const std::string filename = "../plots/plot_" + int_to_string(n, 3) + ".dat";
            tpl.open(filename);
            tpl.flags(std::ios::dec | std::ios::scientific);
            tpl.precision(6);

            tpl << "TITLE = \"Wave Equation 2D\" " << std::endl
                << "VARIABLES = \"x\", \"y\", \"U\", \"V\" " << std::endl;
            tpl << "Zone I = " << Ny << " J = " << Nx << std::endl;
            tpl << "SOLUTIONTIME = " << t << std::endl;

            for (int i = 0; i < Nx; i++)
            {
                for (int j = 0; j < Ny; j++)
                {
                    tpl << x(i) << "\t" << y(j) << "\t" << u(i, j) << "\t" << v(i, j) << std::endl;
                }
            }

            tpl.close();
        }

        n++;
        t += dt;
    }

    return 0;
}
