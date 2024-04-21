// cylindrical coordinate system
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_randist.h>
#include <iostream>
#include <math.h>
#include <vector>
#include "pcmAnalysis.h"
#include <fstream>
#include <unistd.h>
#include <signal.h>
#include <assert.h>

#define dim_x 6
#define dim_z 6

using namespace std;

double c = 340;
double radius = 0.06;
double beta = 2 * M_PI / dim_z;

fstream out;

//capture ctrl + c
void close_sig_handler(int sig){
    cout << "Over" << endl;
    out.close();
    exit(0);
}

double h(const int i, const double rr, const double zz, const double ff)
{
        double loss;
        loss = (sqrt(rr * rr + radius * radius + zz * zz - 2 * radius * rr * cos(i * beta - ff)) - sqrt(rr * rr + zz * zz)) / c;
        return loss;
}

void m_printf(const gsl_matrix *m, size_t dim_i, size_t dim_j)
{
        for (int i = 0; i < dim_i; ++i)
        {
                for (int j = 0; j < dim_j; ++j)
                {
                        if (i == 2){
                                double result = fmod(fmod(gsl_matrix_get(m, i, j), 2 * M_PI) + 2 * M_PI, 2 * M_PI) / (2 * M_PI) * 360;
                                cout << setw(15) << result;
                        }    
                        else
                                cout << setw(15) << gsl_matrix_get(m, i, j);
                }
        }
        cout << endl;
}

//command: doa_locating GCC 0.001 1e-9
int main(int argc,char *argv[])
{
        PcmAnalysis pcm;

        struct sigaction sa;
        sa.sa_handler = close_sig_handler;
        assert(sigaction(SIGINT,&sa,NULL)!=-1);

        out.open("results.txt", ios_base::out);

        // create kalman filter parameters
        double z[dim_z * 1] = {0};
        double z_pred[dim_z * 1] = {0};
        double x_[dim_x * 1] = {0};
        double P_[dim_x * dim_x] = {0};
        double K[dim_x * dim_z] = {0};
        double I[dim_x * dim_x] = {1, 0, 0, 0, 0, 0,
                                   0, 1, 0, 0, 0, 0,
                                   0, 0, 1, 0, 0, 0,
                                   0, 0, 0, 1, 0, 0,
                                   0, 0, 0, 0, 1, 0,
                                   0, 0, 0, 0, 0, 1};
        double H[dim_z * dim_x] = {0};
        double F[dim_x * dim_x] = {1, 0, 0, pcm.getDt(), 0, 0,
                                   0, 1, 0, 0, pcm.getDt(), 0,
                                   0, 0, 1, 0, 0, pcm.getDt(),
                                   0, 0, 0, 1, 0, 0,
                                   0, 0, 0, 0, 1, 0,
                                   0, 0, 0, 0, 0, 1};
        double R_param = stod(string(argv[3]));
        double R[dim_z * dim_z] = {R_param, 0, 0, 0, 0, 0,
                                   0, R_param, 0, 0, 0, 0,
                                   0, 0, R_param, 0, 0, 0,
                                   0, 0, 0, R_param, 0, 0,
                                   0, 0, 0, 0, R_param, 0,
                                   0, 0, 0, 0, 0, R_param};
        double Q_param = stod(string(argv[2]));
        double Q[dim_x * dim_x] = {Q_param, 0, 0, 0, 0, 0,
                                   0, Q_param, 0, 0, 0, 0,
                                   0, 0, Q_param, 0, 0, 0,
                                   0, 0, 0, Q_param, 0, 0,
                                   0, 0, 0, 0, Q_param, 0,
                                   0, 0, 0, 0, 0, Q_param};
        double P[dim_x * dim_x] = {1};
        double x[dim_x * 1] = {0.1,
                               0.1,
                               0,
                               0,
                               0,
                               0};

        gsl_matrix_view MATRIX_z = gsl_matrix_view_array(z, dim_z, 1);
        gsl_matrix_view MATRIX_z_pred = gsl_matrix_view_array(z_pred, dim_z, 1);
        gsl_matrix_view MATRIX_x_ = gsl_matrix_view_array(x_, dim_x, 1);
        gsl_matrix_view MATRIX_P_ = gsl_matrix_view_array(P_, dim_x, dim_x);
        gsl_matrix_view MATRIX_K = gsl_matrix_view_array(K, dim_x, dim_z);
        gsl_matrix_view MATRIX_I = gsl_matrix_view_array(I, dim_x, dim_x);
        gsl_matrix_view MATRIX_H = gsl_matrix_view_array(H, dim_z, dim_x);
        gsl_matrix_view MATRIX_F = gsl_matrix_view_array(F, dim_x, dim_x);
        gsl_matrix_view MATRIX_R = gsl_matrix_view_array(R, dim_z, dim_z);
        gsl_matrix_view MATRIX_Q = gsl_matrix_view_array(Q, dim_x, dim_x);
        gsl_matrix_view MATRIX_P = gsl_matrix_view_array(P, dim_x, dim_x);
        gsl_matrix_view MATRIX_x = gsl_matrix_view_array(x, dim_x, 1);

        double temp_0[dim_x * dim_x] = {0};
        double temp_1[dim_x * dim_z] = {0};
        double temp_2[dim_z * dim_x] = {0};
        double temp_3[dim_z * dim_z] = {0};
        double temp_4[dim_z * dim_z] = {0};
        double temp_5[dim_x * 1] = {0};
        double temp_6[dim_x * dim_x] = {0};

        gsl_matrix_view m_temp_0 = gsl_matrix_view_array(temp_0, dim_x, dim_x);
        gsl_matrix_view m_temp_1 = gsl_matrix_view_array(temp_1, dim_x, dim_z);
        gsl_matrix_view m_temp_2 = gsl_matrix_view_array(temp_2, dim_z, dim_x);
        gsl_matrix_view m_temp_3 = gsl_matrix_view_array(temp_3, dim_z, dim_z);
        gsl_matrix_view m_temp_4 = gsl_matrix_view_array(temp_4, dim_z, dim_z);
        gsl_matrix_view m_temp_5 = gsl_matrix_view_array(temp_5, dim_x, 1);
        gsl_matrix_view m_temp_6 = gsl_matrix_view_array(temp_6, dim_x, dim_x);

        cout << setw(15) << "r" << setw(15) << "z" << setw(15) << "phi" << setw(15) << "r_speed" << setw(15) << "z_speed" << setw(15) << "phi_speed" << endl;
        while (1)
        {
                // generating obs
                double delays[dim_z * 1] = {0};
                if(string(argv[1]) == "CC") pcm.getDelay_CC(delays, dim_z * 1);
                else if(string(argv[1]) == "CPS") pcm.getDelay_CPS(delays, dim_z * 1);
                else if(string(argv[1]) == "GCC") pcm.getDelay_GCC_PHAT(delays, dim_z * 1);
                gsl_matrix_view m_delays = gsl_matrix_view_array(delays, dim_z, 1);
                gsl_matrix_memcpy(&MATRIX_z.matrix, &m_delays.matrix);

                // kalman predict
                gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &MATRIX_F.matrix, &MATRIX_x.matrix, 0.0, &MATRIX_x_.matrix);
                gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &MATRIX_F.matrix, &MATRIX_P.matrix, 0.0, &m_temp_0.matrix);
                gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, &m_temp_0.matrix, &MATRIX_F.matrix, 0.0, &MATRIX_P_.matrix);
                gsl_matrix_add(&MATRIX_P_.matrix, &MATRIX_Q.matrix);
                gsl_matrix_set(&MATRIX_x_.matrix, 0, 0, abs(gsl_matrix_get(&MATRIX_x_.matrix, 0, 0)));
                gsl_matrix_set(&MATRIX_x_.matrix, 1, 0, abs(gsl_matrix_get(&MATRIX_x_.matrix, 1, 0)));
                // for(int num = 0; num < dim_x; num++){
                //         for(int num2 = 0; num2 < dim_x; num2++)
                //                 cout << gsl_matrix_get(&MATRIX_P.matrix, num, num2) << " ";
                //         cout << endl;
                // }  
                // cout << endl << endl;
                
                // kalman update H matrix
                for (int k = 0; k < dim_z; k++)
                {
                        double loss = 0;
                        double rr = gsl_matrix_get(&MATRIX_x_.matrix, 0, 0);
                        double zz = gsl_matrix_get(&MATRIX_x_.matrix, 1, 0);
                        double ff = gsl_matrix_get(&MATRIX_x_.matrix, 2, 0);
                        // rr = 2.5;
                        // zz = 2.5;
                        // ff = M_PI / 3.0;
                        loss = h(k, rr, zz, ff);
                        // cout << loss << " ";
                        gsl_matrix_set(&MATRIX_z_pred.matrix, k, 0, loss);
                        double rr_grad = (-(rr / sqrt(rr * rr + zz * zz)) + (rr - radius * cos(ff - beta * k)) / sqrt(radius * radius + rr * rr + zz * zz - 2.0 * radius * rr * cos(ff - beta * k))) / c;
                        double zz_grad = (zz * (-(1.0 / sqrt(rr * rr + zz * zz)) + 1.0 / sqrt(radius * radius + rr * rr + zz * zz - 2.0 * radius * rr * cos(ff - beta * k)))) / c;
                        double ff_grad = (radius * rr * sin(ff - beta * k)) / (c * sqrt(radius * radius + rr * rr + zz * zz - 2.0 * radius * rr * cos(ff - beta * k)));
                        gsl_matrix_set(&MATRIX_H.matrix, k, 0, rr_grad);
                        gsl_matrix_set(&MATRIX_H.matrix, k, 1, zz_grad);
                        gsl_matrix_set(&MATRIX_H.matrix, k, 2, ff_grad);
                        gsl_matrix_set(&MATRIX_H.matrix, k, 3, rr_grad * pcm.getDt());
                        gsl_matrix_set(&MATRIX_H.matrix, k, 4, zz_grad * pcm.getDt());
                        gsl_matrix_set(&MATRIX_H.matrix, k, 5, ff_grad * pcm.getDt());
                }
                // exit(0);

                // kalman update
                gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, &MATRIX_P_.matrix, &MATRIX_H.matrix, 0.0, &m_temp_1.matrix);
                gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &MATRIX_H.matrix, &MATRIX_P_.matrix, 0.0, &m_temp_2.matrix);
                gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, &m_temp_2.matrix, &MATRIX_H.matrix, 0.0, &m_temp_3.matrix);
                gsl_matrix_add(&m_temp_3.matrix, &MATRIX_R.matrix);
                gsl_permutation *perm = gsl_permutation_alloc(dim_z);
                int sign;
                gsl_linalg_LU_decomp(&m_temp_3.matrix, perm, &sign);
                gsl_linalg_LU_invert(&m_temp_3.matrix, perm, &m_temp_4.matrix);
                gsl_permutation_free(perm);
                gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &m_temp_1.matrix, &m_temp_4.matrix, 0.0, &MATRIX_K.matrix);
                // for(int num = 0; num < dim_z; num++)
                //         cout << "MATRIX_z: " << gsl_matrix_get(&MATRIX_z.matrix, num, 0) << " ";
                // cout << endl;
                // for(int num = 0; num < dim_z; num++)
                //         cout << "MATRIX_z_pred: " << gsl_matrix_get(&MATRIX_z_pred.matrix, num, 0) << " ";
                // cout << endl;
                gsl_matrix_sub(&MATRIX_z.matrix, &MATRIX_z_pred.matrix);
                // for(int num = 0; num < dim_z; num++)
                //         cout << "loss: " << gsl_matrix_get(&MATRIX_z.matrix, num, 0) << " ";
                // cout << endl << endl;
                gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &MATRIX_K.matrix, &MATRIX_z.matrix, 0.0, &m_temp_5.matrix);
                gsl_matrix_add(&m_temp_5.matrix, &MATRIX_x_.matrix);
                gsl_matrix_memcpy(&MATRIX_x.matrix, &m_temp_5.matrix);
                gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &MATRIX_K.matrix, &MATRIX_H.matrix, 0.0, &m_temp_6.matrix);
                double scale_x = -1;
                gsl_matrix_scale(&m_temp_6.matrix, scale_x);                                                                               
                gsl_matrix_add(&m_temp_6.matrix, &MATRIX_I.matrix);
                gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &m_temp_6.matrix, &MATRIX_P_.matrix, 0.0, &MATRIX_P.matrix);
                gsl_matrix_set(&MATRIX_x.matrix, 0, 0, abs(gsl_matrix_get(&MATRIX_x.matrix, 0, 0)));
                gsl_matrix_set(&MATRIX_x.matrix, 1, 0, abs(gsl_matrix_get(&MATRIX_x.matrix, 1, 0)));
                m_printf(&MATRIX_x.matrix, dim_x, 1);
                //output DOA angle result
                double result = fmod(fmod(gsl_matrix_get(&MATRIX_x.matrix, 2, 0), 2 * M_PI) + 2 * M_PI, 2 * M_PI) / (2 * M_PI) * 360;
                out << result << " ";
        }

        out.close();
        return 0;
}