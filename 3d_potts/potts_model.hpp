#ifndef potts_model_hpp
#define potts_model_hpp

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <random>
#include <armadillo>

#include "util.hpp"

using namespace std;

typedef mt19937 RNG;


class Cubic_Lattice {
public:
    RNG rng;

    int L, Nx, Ny, Nz, Ns;
    double J1 = 1.0;
    double J2 = 0.15;
    double J3 = 0.50;

    int sim_type = 1;
    int stack_axis = 0;
    
    vector<vector<double>> J_x, J_y, J_z;
    vector<vector<double>> t_x, t_y, t_z;

    int dim;
    double mu;
    double filling;
    double kT;

    static constexpr int N_nn1 = 6;

    class Site {
    public:
        int idx;
        int x, y, z;

        int state;
        double delta_Es[3];

        Site *nn1[N_nn1];

        Site(void){ };
    } *site;

    arma::sp_mat Hamiltonian;
    arma::mat Density_Mat;

    Cubic_Lattice(int linear_size) {
        L = linear_size;
        Nx = L;
        Ny = L;
        Nz = L;
        Ns = L * L * L;
        dim = Ns;

        site = new Site[Ns];

        Hamiltonian = arma::sp_mat(dim, dim);
        Density_Mat = arma::mat(dim, dim);

        t_x = {
            {1.0, -0.5, -0.5},
            {-0.5, -0.7, -0.7},
            {-0.5, -0.7, -0.7}
        };
        t_y = {
            {-0.7, -0.5, -0.7},
            {-0.5, 1.0, -0.5},
            {-0.7, -0.5, -0.7}
        };
        t_z = {
            {-0.7, -0.7, -0.5},
            {-0.7, -0.7, -0.5},
            {-0.5, -0.5, 1.0}
        };
        
        J_x = { 
            {J1, 0, J2}, 
            {0, 0, 0}, 
            {J2, 0, 0} 
        };
        J_y = { 
            {0, J2, 0}, 
            {J2, J1, 0}, 
            {0, 0, 0} 
        };
        J_z = { 
            {0, 0, 0}, 
            {0, 0, J2}, 
            {0, J2, J1} 
        };

        std::random_device seed;
        rng = RNG(seed());

        init_lattice();
    }

    ~Cubic_Lattice(void) {
        delete [] site;
    }

    inline int idx_site(int x, int y, int z) {
        return (x * L * L) + (y * L) + z;
    }

    void init_lattice(void);
    void init_random(void);

    arma::sp_mat build_Hamiltonian(void);

    void compute_fermi_level(arma::vec & eigE);
    double compute_energy_silent(void);
    double classical_site_energy(int i, int state_i);

    void compute_delta_E(void);

    void save_dataset(string const filename);
};

#endif
