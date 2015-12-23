// (C) Copyright 2009, Chong Wang, David Blei and Li Fei-Fei

// written by Chong Wang, chongw@cs.princeton.edu

// This file is part of slda.

// slda is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your
// option) any later version.

// slda is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// for more details.

// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA

#ifndef SLDA_H
#define SLDA_H
#include "settings.h"
#include "corpus.h"

typedef struct {
    double * z_bar_m;
    double * z_bar_var;
} z_stat;

typedef struct {
    double ** word_ss;
    double * word_total_ss;
    double *** pi_stat;
    double ** pi_total_stat;
    int num_docs;
    z_stat * z_bar;
} suffstats;

class slda
{
public:
    slda();
    ~slda();
    void free_model();
    void init(double alpha_, double tau_, double omega_, int num_topics_, const corpus * c, const settings * setting, double init_seed, int batch_size);
    void v_em(corpus * c, const char * start, const char * directory, const settings * setting);

    void save_model(char * filename);
    void save_model_text(const char * filename);
    void load_model(const char * model_filename, const settings * setting);
    void infer_only(corpus * c, const char * directory);

    void initialize_pi();
    suffstats * new_suffstats(int num_docs);
    void free_suffstats(suffstats * ss);
    void zero_initialize_ss(suffstats * ss, int ZETA_XI);
    void random_initialize_ss(suffstats * ss, corpus * c);
    void corpus_initialize_ss(suffstats* ss, corpus * c, double** lambda);
    void load_model_initialize_ss(suffstats* ss, corpus * c);

    double zeta_xi_estimation(suffstats * ss, int eta_update, int step, int sample_size);
    void mle(suffstats * ss, int eta_update, double ** lambda, corpus * c);

    double doc_e_step(document* doc, double* gamma, double** phi, double * lambda, suffstats * ss, int eta_update);

    double lda_inference(document* doc, double* var_gamma, double** phi);
    double lda_compute_likelihood(document* doc, double** phi, double* var_gamma);
    double slda_inference(document* doc, double* var_gamma, double** phi, double * lambda);
    double slda_compute_likelihood(document* doc, double** phi, double* var_gamma, double* lambda);
    double zeta_xi_likelihood();

    void save_gamma(char* filename, double** gamma, int num_docs);
    void write_word_assignment(FILE* f, document* doc, double** phi);


public:
    double alpha; // the parameter for the dirichlet
    double tau;
    double omega;
    double penalty;
    double lambda_smoother;
    double var_converged;
    double em_converged;
    double seed;
    double forgetting_rate;

    int batch_size;
    int delay;
    int num_topics;
    int num_classes;
    int size_vocab;
    int num_docs;
    int num_annot;
    int pi_estimation;
    int var_max_iter;
    int em_max_iter;
   
    int * docs_by_class;

    char * pi_filename;

    double *** true_pi;
    double *** pi_est;
    double *** exp_log_pi;
    double *** xi;
    double ** zeta;
    double ** exp_log_beta; //the log of the topic distribution
    double ** eta; //softmax regression, in general, there are num_classes-1 etas, we don't need a intercept here, since \sum_i \bar{z_i} = 1

};

#endif // SLDA_H

