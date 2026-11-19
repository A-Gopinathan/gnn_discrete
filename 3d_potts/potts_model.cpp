#include "potts_model.hpp"
#include "util.hpp"

#include <stdexcept>

using namespace std;

void Cubic_Lattice::init_lattice(void) {

    for (int x = 0; x < Nx; x++) {
        for (int y = 0; y < Ny; y++) {
            for (int z = 0; z < Nz; z++) {
                int i = idx_site(x, y, z);
                site[i].idx = i;
                site[i].x = x;
                site[i].y = y;
                site[i].z = z;
            }
        }
    }

    for (int i = 0; i < Ns; i++) {
        int x = site[i].x;
        int y = site[i].y;
        int z = site[i].z;

        site[i].nn1[0] = &site[idx_site(mod(x + 1, Nx), y, z)];
        site[i].nn1[1] = &site[idx_site(mod(x - 1, Nx), y, z)];

        site[i].nn1[2] = &site[idx_site(x, mod(y + 1, Ny), z)];
        site[i].nn1[3] = &site[idx_site(x, mod(y - 1, Ny), z)];

        site[i].nn1[4] = &site[idx_site(x, y, mod(z + 1, Nz))];
        site[i].nn1[5] = &site[idx_site(x, y, mod(z - 1, Nz))];
    }
}

void Cubic_Lattice::init_random(void) {
    uniform_int_distribution<int> color_dist(0, 2);
    for (int i = 0; i < Ns; i++) {
        site[i].state = color_dist(rng);
    }
}

arma::sp_mat Cubic_Lattice::build_Hamiltonian(void) {
    arma::sp_mat H(dim, dim);

    for(int i=0; i<Ns; i++) {
        int state_i = site[i].state;
        for(int k=0; k<N_nn1; k++) {
            int j = site[i].nn1[k]->idx;
            int state_j = site[i].nn1[k]->state;

            double t_val = 0.0;

            if (k <= 1) {
                t_val = t_x[state_i][state_j];
            }
            else if (k <= 3) {
                t_val = t_y[state_i][state_j];
            }
            else {
                t_val = t_z[state_i][state_j];
            }

            H(i, j) = -t_val;
        }
    }

    return H;
}

void Cubic_Lattice::compute_fermi_level(arma::vec & eigE) {
    double x1 = eigE(0);
    double x2 = eigE(eigE.size()-1);

    int max_bisection = 500;
    double eps_bisection = 1.e-12;

    int iter = 0;
    while(iter < max_bisection && fabs(x2 - x1) > eps_bisection) {
        double xm = 0.5*(x1 + x2);
        double density = 0;
        for(int i=0; i<eigE.size(); i++) {
            density += fermi_density(eigE(i), kT, xm);
        }
        density /= ((double) dim);

        if(density <= filling) x1 = xm;
        else x2 = xm;

        iter++;
    }
    mu = 0.5*(x1 + x2);
}

double Cubic_Lattice::compute_energy_silent(void) {
    Hamiltonian = build_Hamiltonian();

    arma::mat Hm(Hamiltonian);
    arma::vec eigval(dim);

    if (!arma::eig_sym(eigval, Hm)) {
        throw runtime_error("Hamiltonian eigendecomposition failed");
    }

    compute_fermi_level(eigval);

    arma::vec fd_factor(dim);
    for(int i=0; i<dim; i++) fd_factor(i) = fermi_density(eigval(i), kT, mu);

    double sum = 0;
    for(int i=0; i<dim; i++) {
        sum += eigval(i) * fd_factor(i);
    }


    return sum;
}

double Cubic_Lattice::classical_site_energy(int i, int state_i) {
    double E_c = 0;

    // 1. Determine effective J3 penalties based on Sim Type
    double eff_Jx_J3 = (sim_type == 2 || (sim_type == 3 && stack_axis == 1)) ? J3 : 0.50;
    double eff_Jy_J3 = (sim_type == 2 || (sim_type == 3 && stack_axis == 2)) ? J3 : 0.50;
    double eff_Jz_J3 = (sim_type == 2 || (sim_type == 3 && stack_axis == 3)) ? J3 : 0.50;

    // 2. Loop through neighbors and apply the correct classical penalty
    for(int k = 0; k < N_nn1; k++) {
        int state_j = site[i].nn1[k]->state;

        if (k <= 1) { // x-bond
            // If it's Y-Y or Z-Z on an X-bond, apply the dynamic J3 penalty
            if (state_i == state_j && state_i != 0) E_c += eff_Jx_J3;
            else E_c += J_x[state_i][state_j];
        }
        else if (k <= 3) { // y-bond
            // If it's X-X or Z-Z on a Y-bond, apply the dynamic J3 penalty
            if (state_i == state_j && state_i != 1) E_c += eff_Jy_J3;
            else E_c += J_y[state_i][state_j];
        }
        else { // z-bond
            // If it's X-X or Y-Y on a Z-bond, apply the dynamic J3 penalty
            if (state_i == state_j && state_i != 2) E_c += eff_Jz_J3;
            else E_c += J_z[state_i][state_j];
        }
    }

    // 3. Apply On-Site Tax to exhaust the stretched orbital (Sim Type 3 only)
    double on_site_tax = 0.0;
    if (sim_type == 3) {
        // FIXED: Changed state_val to state_i
        if (stack_axis == 1 && state_i == 0) on_site_tax = 0.5;
        if (stack_axis == 2 && state_i == 1) on_site_tax = 0.5;
        if (stack_axis == 3 && state_i == 2) on_site_tax = 0.5;
    }

    E_c += on_site_tax;
    return E_c;
}


void Cubic_Lattice::compute_delta_E(void) {
    double E0 = compute_energy_silent();

    for (int i = 0; i < Ns; i++) {
        int original_state = site[i].state;

        for (int c = 0; c < 3; c++) {
            if (c == original_state) {
                site[i].delta_Es[c] = 0.0;
            }
            else {
                site[i].state = c;
                double E_new = compute_energy_silent();
                site[i].delta_Es[c] = E_new - E0;
            }
        }
        site[i].state = original_state;
    }
}

void Cubic_Lattice::save_dataset(string const filename) {
    std::ofstream fs;
    fs.open(filename.c_str(), ios::out);
    fs.precision(10);

    fs << "x,y,z,state,delta_E_R,delta_E_G,delta_E_B\n";

    for (int i = 0; i < Ns; i++) {

        fs << site[i].x << ","
        << site[i].y << ","
        << site[i].z << ","
        << site[i].state << ","
        << site[i].delta_Es[0] << ","
        << site[i].delta_Es[1] << ","
        << site[i].delta_Es[2] << "\n";
    }
    fs.close();
}
