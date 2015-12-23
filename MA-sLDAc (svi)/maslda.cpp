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

#include "maslda.h"
#include <time.h>
#include "utils.h"
#include "assert.h"
#include "opt.h"
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <iostream>
#include <fstream>

const int NUM_INIT = 50;
const int LAG = 10;
const int LDA_INIT_MAX = 0;
const int MSTEP_MAX_ITER = 50;

slda::slda()
{
    //ctor
    alpha = 1.0;
    num_topics = 0;
    num_classes = 0;
    size_vocab = 0;
    num_annot = 0;
    num_docs = 0;

    exp_log_beta = NULL;
    eta = NULL;
}

slda::~slda()
{
    free_model();
}

/*
 * init the model
 */

void slda::init(double alpha_, double tau_, double omega_, int num_topics_,
                const corpus * c, const settings * setting, double init_seed, int batch_size_)
{
    alpha = alpha_;
    omega = omega_;
    tau = tau_;
    num_topics = num_topics_;
    size_vocab = c->size_vocab;
    num_classes = c->num_classes;
    num_annot = c->num_annot;
    num_docs = c->num_docs;
    seed = init_seed;
    batch_size = batch_size_;

    pi_filename = new char[1000];
    memcpy(pi_filename, setting->PI_FILE, sizeof setting->PI_FILE);
    pi_estimation = setting->ESTIMATE_PI;
    lambda_smoother = setting->LAMBDA_SMOOTHER;
    penalty = setting->PENALTY;
    delay = setting->DELAY;
    forgetting_rate = setting->FORGETTING_RATE;

    var_converged = setting->VAR_CONVERGED;
    var_max_iter = setting->VAR_MAX_ITER;
    em_converged = setting->EM_CONVERGED;
    em_max_iter = setting->EM_MAX_ITER;

    docs_by_class = new int[num_classes];
    memset(docs_by_class, 0, sizeof(int)*num_classes);

    for(int d = 0; d < num_docs; d++)
        docs_by_class[c->docs[d]->label]++;

    exp_log_beta = new double * [num_topics];
    zeta = new double * [num_topics];
    for (int k = 0; k < num_topics; k++)
    {
        exp_log_beta[k] = new double [size_vocab];
        zeta[k] = new double [size_vocab]; 
        memset(exp_log_beta[k], 0, sizeof(double)*size_vocab);
    }

    //no need to train slda if we only have on class
    if (num_classes > 1)
    {
        eta = new double * [num_classes-1];
        for (int i = 0; i < num_classes-1; i ++)
        {
            eta[i] = new double [num_topics];
            memset(eta[i], 0, sizeof(double)*num_topics);
        }
    }

    exp_log_pi = new double ** [num_annot];
    xi = new double ** [num_annot];
    pi_est = new double ** [num_annot];
    for (int a = 0; a < num_annot; a ++)
    {
        exp_log_pi[a] = new double * [num_classes];
        xi[a] = new double * [num_classes];
        pi_est[a] = new double * [num_classes];
        for (int c = 0; c < num_classes; c ++)
        {
            exp_log_pi[a][c] = new double [num_classes];
            memset(exp_log_pi[a][c], 0, sizeof(double)*num_classes);

            xi[a][c] = new double [num_classes];
            pi_est[a][c] = new double [num_classes];
        }
    }
}

/*
 * free the model
 */

void slda::free_model()
{
    if (exp_log_beta != NULL)
    {
        for (int k = 0; k < num_topics; k++)
        {
            delete [] exp_log_beta[k];
        }
        delete [] exp_log_beta;
        exp_log_beta = NULL;
    }
    if (eta != NULL)
    {
        for (int i = 0; i < num_classes-1; i ++)
        {
            delete [] eta[i];
        }
        delete [] eta;
        eta = NULL;
    }
}

/*
 * save the model in the binary format
 */

void slda::save_model(char * filename)
{
    printf("SAVING MODEL...\n");
    printf("FILENAME=%s\n", filename);
    FILE * file = NULL;
    file = fopen(filename, "wb");
    if (file != NULL) {
        fwrite(&alpha, sizeof (double), 1, file);
        fwrite(&tau, sizeof (double), 1, file);
        fwrite(&omega, sizeof (double), 1, file);
        fwrite(&num_topics, sizeof (int), 1, file);
        fwrite(&size_vocab, sizeof (int), 1, file);
        fwrite(&num_classes, sizeof (int), 1, file);

        for (int k = 0; k < num_topics; k++)
        {
            fwrite(exp_log_beta[k], sizeof(double), size_vocab, file);
        }
        if (num_classes > 1)
        {
            for (int i = 0; i < num_classes-1; i ++)
            {
                fwrite(eta[i], sizeof(double), num_topics, file);
            }
        }

        fflush(file);
        fclose(file);
    }
}

/*
 * load the model in the binary format
 */

void slda::load_model(const char * filename, const settings * setting)
{
    FILE * file = NULL;
    file = fopen(filename, "rb");
    fread(&alpha, sizeof (double), 1, file);
    fread(&tau, sizeof (double), 1, file);
    fread(&omega, sizeof (double), 1, file);
    fread(&num_topics, sizeof (int), 1, file);
    fread(&size_vocab, sizeof (int), 1, file);
    fread(&num_classes, sizeof (int), 1, file);

    pi_filename = new char[1000];
    memcpy(pi_filename, setting->PI_FILE, sizeof setting->PI_FILE);
    pi_estimation = setting->ESTIMATE_PI;
    lambda_smoother = setting->LAMBDA_SMOOTHER;
    penalty = setting->PENALTY;

    var_converged = setting->VAR_CONVERGED;
    var_max_iter = setting->VAR_MAX_ITER;
    em_converged = setting->EM_CONVERGED;
    em_max_iter = setting->EM_MAX_ITER;

    exp_log_beta = new double * [num_topics];
    for (int k = 0; k < num_topics; k++)
    {
        exp_log_beta[k] = new double [size_vocab];
        fread(exp_log_beta[k], sizeof(double), size_vocab, file);
    }
    if (num_classes > 1)
    {
        eta = new double * [num_classes-1];
        for (int i = 0; i < num_classes-1; i ++)
        {
            eta[i] = new double [num_topics];
            fread(eta[i], sizeof(double), num_topics, file);
        }
    }

    fflush(file);
    fclose(file);
}

/*
 * save the model in the text format
 */

void slda::save_model_text(const char * filename)
{
    FILE * file = NULL;
    file = fopen(filename, "w");
    if (file != NULL) {
        fprintf(file, "alpha: %lf\n", alpha);
        fprintf(file, "tau: %lf\n", tau);
        fprintf(file, "omega: %lf\n", omega);
        fprintf(file, "number of topics: %d\n", num_topics);
        fprintf(file, "size of vocab: %d\n", size_vocab);
        fprintf(file, "number of classes: %d\n", num_classes);

        fprintf(file, "betas: \n"); // in log space
        for (int k = 0; k < num_topics; k++)
        {
            for (int j = 0; j < size_vocab; j ++)
            {
                fprintf(file, "%lf ", exp_log_beta[k][j]);
            }
            fprintf(file, "\n");
        }
        if (num_classes > 1)
        {
            fprintf(file, "etas: \n");
            for (int i = 0; i < num_classes-1; i ++)
            {
                for (int j = 0; j < num_topics; j ++)
                {
                    fprintf(file, "%lf ", eta[i][j]);
                }
                fprintf(file, "\n");
            }
        }

        fflush(file);
        fclose(file);
    }
}

/*
 * create the data structure for sufficient statistic 
 */

suffstats * slda::new_suffstats(int num_docs)
{
    suffstats * ss = new suffstats;
    ss->num_docs = num_docs;
    ss->word_total_ss = new double [num_topics];
    memset(ss->word_total_ss, 0, sizeof(double)*num_topics);
    ss->word_ss = new double * [num_topics];
    for (int k = 0; k < num_topics; k ++)
    {
        ss->word_ss[k] = new double [size_vocab];
        memset(ss->word_ss[k], 0, sizeof(double)*size_vocab);
    }

    int num_var_entries = num_topics*(num_topics+1)/2; 
    ss->z_bar =  new z_stat [num_docs];
    for (int d = 0; d < num_docs; d ++)
    {
        ss->z_bar[d].z_bar_m = new double [num_topics];
        ss->z_bar[d].z_bar_var = new double [num_var_entries];
        memset(ss->z_bar[d].z_bar_m, 0, sizeof(double)*num_topics);
        memset(ss->z_bar[d].z_bar_var, 0, sizeof(double)*num_var_entries);
    }
    ss->pi_stat = new double ** [num_annot];
    ss->pi_total_stat = new double * [num_annot];
    for (int a = 0; a < num_annot; a ++)
    {
        ss->pi_stat[a] = new double * [num_classes];
        for (int c = 0; c < num_classes; c ++)
        {
            ss->pi_stat[a][c] = new double [num_classes];
            memset(ss->pi_stat[a][c], 0, sizeof(double)*num_classes);
        }
        ss->pi_total_stat[a] = new double [num_classes];
        memset(ss->pi_total_stat[a], 0, sizeof(double)*num_classes);
    }

    return(ss);
}

void slda::initialize_pi()
{
    if(pi_estimation)
    {
        for (int a = 0; a < num_annot; a ++)
            for (int c = 0; c < num_classes; c ++)
                for(int l = 0; l < num_classes; l++)
                    if (c==l)
                        exp_log_pi[a][c][l] = log(0.97);
                    else
                        exp_log_pi[a][c][l] = log(0.01);
    }
    else
    {
        for(int n = 0; n < num_annot; n++)
            for(int c = 0; c < num_classes; c++)
                for (int l = 0; l < num_classes; l++)
                    exp_log_pi[n][c][l] = log(true_pi[n][c][l]);
    }
}

/*
 * initialize the sufficient statistics with zeros
 */

void slda::zero_initialize_ss(suffstats * ss, int flag)
{

    int a, c;

    if(!flag)
    {
        int num_var_entries = num_topics*(num_topics+1)/2;
        for (int d = 0; d < ss->num_docs; d ++)
        {
            memset(ss->z_bar[d].z_bar_m, 0, sizeof(double)*num_topics);
            memset(ss->z_bar[d].z_bar_var, 0, sizeof(double)*num_var_entries);
        }
        ss->num_docs = 0;
    }

    memset(ss->word_total_ss, 0, sizeof(double)*num_topics);
    for (int k = 0; k < num_topics; k ++)
        memset(ss->word_ss[k], 0, sizeof(double)*size_vocab);

    if(pi_estimation)
        for (a = 0; a < num_annot; a ++)
	{
            memset(ss->pi_total_stat[a], 0, sizeof(double)*num_classes);
            for (c = 0; c < num_classes; c ++)
                memset(ss->pi_stat[a][c], 0, sizeof(double)*num_classes);
	}
}

/*
 * initialize the sufficient statistics with random numbers 
 */

void slda::random_initialize_ss(suffstats * ss, corpus* c)
{
    int num_docs = ss->num_docs;
    gsl_rng * rng = gsl_rng_alloc(gsl_rng_taus);
    gsl_rng_set(rng, (long) seed);  
    int k, w, d, j, a, l, c_, idx;
    for (k = 0; k < num_topics; k++)
    {
        for (w = 0; w < size_vocab; w++)
        {
            ss->word_ss[k][w] = 1.0/size_vocab + 0.1*gsl_rng_uniform(rng);
            ss->word_total_ss[k] += ss->word_ss[k][w];
        }
    }

    for (d = 0; d < num_docs; d ++)
    {
        document * doc = c->docs[d];

        double total = 0.0;
        for (k = 0; k < num_topics; k ++)
        {
            ss->z_bar[d].z_bar_m[k] = gsl_rng_uniform(rng);
            total += ss->z_bar[d].z_bar_m[k];
        }

        for (k = 0; k < num_topics; k ++)
        {
            ss->z_bar[d].z_bar_m[k] /= total; 
        }

        for (k = 0; k < num_topics; k ++)
        {
            for (j = k; j < num_topics; j ++)
            {
                idx = map_idx(k, j, num_topics);

                if (j == k)
                    ss->z_bar[d].z_bar_var[idx] = ss->z_bar[d].z_bar_m[k] / (double)(doc->total);
                else
                    ss->z_bar[d].z_bar_var[idx] = 0.0;

                ss->z_bar[d].z_bar_var[idx] -=
                    ss->z_bar[d].z_bar_m[k] * ss->z_bar[d].z_bar_m[j] / (double)(doc->total); 
            }
        }
    }
    if(pi_estimation)
        for (a = 0; a < num_annot; a ++)
            for (c_ = 0; c_ < num_classes; c_ ++)
                for(l = 0; l< num_classes; l ++)
                {
                    ss->pi_stat[a][c_][l] = 1.0/num_classes +  0.1*gsl_rng_uniform(rng);
                    ss->pi_total_stat[a][c_] +=  ss->pi_stat[a][c_][l];
                }

    gsl_rng_free(rng);
}

void slda::corpus_initialize_ss(suffstats* ss, corpus* c, double** lambda)
{
    int num_docs = ss->num_docs;
    gsl_rng * rng = gsl_rng_alloc(gsl_rng_taus);
    int * major_voting = new int[num_docs];
    int * voting = new int[num_classes];
    gsl_rng_set(rng, (long) seed);
    int k, n, d, j, l, a, c_, idx, i, w, majority_class, majority;

    for (k = 0; k < num_topics; k++)
    {
        for (i = 0; i < NUM_INIT; i++)
        {
            d = (int)(floor(gsl_rng_uniform(rng) * num_docs));
            //printf("initialized with document %d\n", d);
            document * doc = c->docs[d];
            for (n = 0; n < doc->length; n++)
                ss->word_ss[k][doc->words[n]] += doc->counts[n];
        }
        for (w = 0; w < size_vocab; w++)
        {
            ss->word_ss[k][w] = 2*ss->word_ss[k][w] + 5 + gsl_rng_uniform(rng);
            ss->word_total_ss[k] = ss->word_total_ss[k] + ss->word_ss[k][w];
        }
    }

    for (d = 0; d < num_docs; d ++)
    {
        document * doc = c->docs[d];
        double total = 0.0;
        for (k = 0; k < num_topics; k ++)
        {
            ss->z_bar[d].z_bar_m[k] = gsl_rng_uniform(rng);
            total += ss->z_bar[d].z_bar_m[k];
        }
        for (k = 0; k < num_topics; k ++)
        {
            ss->z_bar[d].z_bar_m[k] /= total;
        }
        for (k = 0; k < num_topics; k ++)
        {
            for (j = k; j < num_topics; j ++)
            {
                idx = map_idx(k, j, num_topics);
                if (j == k)
                    ss->z_bar[d].z_bar_var[idx] = ss->z_bar[d].z_bar_m[k] / (double)(doc->total);
                else
                    ss->z_bar[d].z_bar_var[idx] = 0.0;

                ss->z_bar[d].z_bar_var[idx] -=
                    ss->z_bar[d].z_bar_m[k] * ss->z_bar[d].z_bar_m[j] / (double)(doc->total);
            }
        }

        if(pi_estimation)
        {
			float total_votes = 0.0;
            memset(voting, 0, sizeof(int)*num_classes);

            for(a = 0; a < num_annot; a++) 
                if (doc->answers[a] >= 0) {
					total_votes += 1;
                    voting[doc->answers[a]]++;
				}

            majority = voting[0];
            majority_class = 0;

            for(c_ = 1; c_ < num_classes; c_++)
			{
				lambda[d][c_] = (voting[c_]+0.0001) / (total_votes + 0.0001*num_classes);
                if(voting[c_]>majority)
                {
                    majority = voting[c_];
                    majority_class = c_;
                }
			}
            major_voting[d] = majority_class;
        }
    }

	for (k = 0; k < num_topics; k++)
	    for (w = 0; w < size_vocab; w++)
        {
            if (ss->word_ss[k][w] > 0) {
                exp_log_beta[k][w] = digamma(tau + ss->word_ss[k][w]) - digamma(tau*size_vocab + ss->word_total_ss[k]);
                zeta[k][w] = tau + ss->word_ss[k][w];
            }
            else {
                exp_log_beta[k][w] = -100.0;
                zeta[k][w] = tau;
            }
        }

    if(pi_estimation)
    {
        for (a = 0; a < num_annot; a ++)
            for (d = 0; d < num_docs; d ++)
            {
                if (c->docs[d]->answers[a] >= 0) 
                {
                    ss->pi_stat[a][major_voting[d]][c->docs[d]->answers[a]]++;
                }
            }

        for (a = 0; a < num_annot; a ++) 
        {
            for (c_ = 0; c_ < num_classes; c_ ++) 
            {
                ss->pi_total_stat[a][c_] = 0;
                for(l = 0; l< num_classes; l ++)
                {
                    ss->pi_total_stat[a][c_] +=  ss->pi_stat[a][c_][l];
                }
				for (l = 0; l < num_classes; l++)
				{ 
                    exp_log_pi[a][c_][l] = digamma(omega + ss->pi_stat[a][c_][l]) - digamma(omega*num_classes + ss->pi_total_stat[a][c_]);
                    xi[a][c_][l] = omega + ss->pi_stat[a][c_][l];
                    pi_est[a][c_][l] = (omega + ss->pi_stat[a][c_][l]) / (omega*num_classes + ss->pi_total_stat[a][c_]);
                }
            }
        }
    }
}

void slda::load_model_initialize_ss(suffstats* ss, corpus * c)
{
    int num_docs = ss->num_docs;                                                                         
    for (int d = 0; d < num_docs; d ++)       
       document * doc = c->docs[d];
}

void slda::free_suffstats(suffstats * ss)
{
    for (int a = 0; a < num_annot; a ++)
    {
        for (int c = 0; c < num_classes; c ++)
            delete [] ss->pi_stat[a][c];

        delete [] ss->pi_stat[a];
        delete [] ss->pi_total_stat[a];
    }
    delete [] ss->pi_total_stat;
    delete [] ss->pi_stat;
    delete [] ss->word_total_ss;
    for (int k = 0; k < num_topics; k ++)
        delete [] ss->word_ss[k];
    delete [] ss->word_ss;
    for (int d = 0; d < ss->num_docs; d ++)
    {
        delete [] ss->z_bar[d].z_bar_m;
        delete [] ss->z_bar[d].z_bar_var;
    }
    delete [] ss->z_bar;
    delete ss;
}

void slda::v_em(corpus * c, const char * start, const char * directory, const settings * setting)
{
    char filename[1000];
    char pi_filename2[1000];
    char m_filename[1000];
    char mt_filename[1000];
    char gamma_filename[1000];
    char w_filename[1000];
    char rmse_filename[1000];
    char lik_filename[1000];
    char lambda_filename[1000];
    char lambda_acc_filename[1000];
    int max_length = c->max_corpus_length();
    double **var_gamma, **phi, **lambda;
    double likelihood, likelihood_old = 0, converged = 1;
    int d, n, i;
    double rmse = -1; 
   
    // allocate variational parameters
    var_gamma = new double * [c->num_docs];
    for (d = 0; d < c->num_docs; d++)
        var_gamma[d] = new double [num_topics];

    phi = new double * [max_length];
    for (n = 0; n < max_length; n++)
        phi[n] = new double [num_topics];

    lambda = new double * [c->num_docs];
    for (n = 0; n < c->num_docs; n++)
        lambda[n] = new double [num_classes];       

    for(int d = 0; d < c->num_docs; d++)
        for(int l = 0; l < num_classes; l++)
            lambda[d][l] = 1/(double) num_classes;

    printf("initializing ...\n");
    suffstats * ss = new_suffstats(c->num_docs);
    FILE * rmse_file = NULL;
    sprintf(rmse_filename, "%s/rmse.txt", directory);
    rmse_file = fopen(rmse_filename, "w");
    FILE * fileptr = NULL;
    fileptr = fopen(pi_filename, "r");
    if(fileptr!=NULL)
    {   
        true_pi = new double ** [num_annot];
        for (int a = 0; a < num_annot; a ++)
        {
            true_pi[a] = new double * [num_classes];
            for (int c = 0; c < num_classes; c ++)
                true_pi[a][c] = new double [num_classes];
        }

        double prob_ann;

        for(int n = 0; n < num_annot; n++)
            for(int c = 0; c < num_classes; c++)
                for (int l = 0; l < num_classes; l++)
                {   
                    fscanf(fileptr, "%lf", &prob_ann);
                    true_pi[n][c][l] = prob_ann;
                }
    }
    fclose(fileptr);

    if (strcmp(start, "seeded") == 0)
    {
        corpus_initialize_ss(ss, c, lambda);
    }
    else if (strcmp(start, "random") == 0)
    {
        random_initialize_ss(ss, c);
        rmse = zeta_xi_estimation(ss, 0, 0, num_docs);
        mle(ss, 0, lambda, c);
	    initialize_pi();
    }
    else
    {
        load_model(start, setting);
        load_model_initialize_ss(ss, c);
    }

    FILE * likelihood_file = NULL;
    sprintf(lik_filename, "%s/likelihood.dat", directory);
    likelihood_file = fopen(lik_filename, "w");
    
    FILE * lambda_acc_file = NULL;
    sprintf(lambda_acc_filename, "%s/lambdas_accuracies.dat", directory);
    lambda_acc_file = fopen(lambda_acc_filename, "w");

    int ETA_UPDATE = 0;
    int step = 0;

    i = 0;
    while (((converged < 0) || (converged > em_converged) || (i <= LDA_INIT_MAX+2)) && (i <= em_max_iter))
    {
        printf("**** em iteration %d ****\n", ++i);
      
        likelihood = 0;
      
        zero_initialize_ss(ss, 0);
      
        if (i > LDA_INIT_MAX) 
            ETA_UPDATE = 1;

        int * accuracy_count = new int [num_classes];
        double * accuracy_prob = new double [num_classes];
        double * accuracy_prob_true = new double [num_classes];

        memset(accuracy_count, 0, sizeof(int)*num_classes); 
        memset(accuracy_prob, 0, sizeof(double)*num_classes);
        memset(accuracy_prob_true, 0, sizeof(double)*num_classes);

        double accuracy_counts = 0;
        double accuracy_prob_trues = 0;
        double max_lambda;
        int max_lambda_pos = 0;

        // e-step
        printf("**** e-step ****\n");
		int last_split_point = -1;
        for (d = 0; d < c->num_docs; d++)
        {   
            if (((d+1) % batch_size) == 0) 
                printf("document %d\t| ", d+1);

            likelihood += doc_e_step(c->docs[d], var_gamma[d], phi, lambda[d], ss, ETA_UPDATE);

            max_lambda = lambda[d][0];
            max_lambda_pos = 0;
            for(int x = 1; x < num_classes; x++)
                if(lambda[d][x] > max_lambda)
                {
                    max_lambda = lambda[d][x];
                    max_lambda_pos = x;
                }

            if(max_lambda_pos == c->docs[d]->label)
                accuracy_count[c->docs[d]->label]++;

            accuracy_prob[c->docs[d]->label]+=max_lambda;
            accuracy_prob_true[c->docs[d]->label]+=lambda[d][c->docs[d]->label]*max_lambda;

			if ((d+1) % batch_size == 0)
            { 
                // update global variational parameters
                zeta_xi_estimation(ss, ETA_UPDATE, step++, d-last_split_point);
				last_split_point = d;

                // re-initilize suff stats to zero
                zero_initialize_ss(ss, 1);
            }
        }

		if ((d-1) > last_split_point) {
			rmse = zeta_xi_estimation(ss, ETA_UPDATE, step++, d-last_split_point);
            if (rmse_file != NULL) {
        	    fprintf(rmse_file, "%.4f\n", rmse);
                fflush(rmse_file);
            }
		}

        likelihood += zeta_xi_likelihood();
        printf("Likelihood: %f\n", likelihood);

        if (c->docs[0]->label != -1) 
        {
            printf("\tLambdas accuracy count: \tLambdas accuracy prob:\n");
            for(int c_ = 0; c_ < num_classes; c_++)
            {
                printf("[%d]\t\t%f\t\t\t%f\n", c_, accuracy_count[c_]/(double)docs_by_class[c_], accuracy_prob_true[c_]/accuracy_prob[c_]);
                accuracy_counts += accuracy_count[c_]/(double)c->num_docs;
                accuracy_prob_trues += accuracy_prob_true[c_]/(double)accuracy_prob[c_] * docs_by_class[c_]/(double)c->num_docs;
            }
            printf("[ALL]\t\t%f\t\t\t%f\n", accuracy_counts, accuracy_prob_trues);
            fprintf(lambda_acc_file, "%f %f\n", accuracy_counts, accuracy_prob_trues);
            fflush(lambda_acc_file);
        }

        // m-step
        printf("**** m-step ****\n");
        mle(ss, ETA_UPDATE, lambda, c);

        // check for convergence
        converged = fabs((likelihood_old - likelihood) / (likelihood_old));
        printf("converged: %f\n", converged);
        //if (converged < 0) VAR_MAX_ITER = VAR_MAX_ITER * 2;
        likelihood_old = likelihood;

        // output model and likelihood
        fprintf(likelihood_file, "%10.10f\t%5.5e\n", likelihood, converged);
        fflush(likelihood_file);
        if ((i % LAG) == 0)
        {
            sprintf(m_filename, "%s/%03d.model", directory, i);
            save_model(m_filename);
            sprintf(mt_filename, "%s/%03d.model.text", directory, i);
            save_model_text(mt_filename);
            sprintf(gamma_filename, "%s/%03d.gamma", directory, i);
            save_gamma(gamma_filename, var_gamma, c->num_docs);
        }
    }

    if (rmse_file != NULL) {
        fflush(rmse_file);
        fclose(rmse_file);
    }

    // output the final model
    sprintf(m_filename, "%s/final.model", directory);
    save_model(m_filename);
    sprintf(mt_filename, "%s/final.model.text", directory);
    save_model_text(mt_filename);
    sprintf(gamma_filename, "%s/final.gamma", directory);
    save_gamma(gamma_filename, var_gamma, c->num_docs);

    fclose(likelihood_file);
    fclose(lambda_acc_file);

    FILE * w_asgn_file = NULL;
    sprintf(w_filename, "%s/word-assignments.dat", directory);
    w_asgn_file = fopen(w_filename, "w");

    for (d = 0; d < c->num_docs; d ++)
    {
        //final inference
        if ((d % 1000) == 0) printf("final e step document %d\n", d);
        likelihood += slda_inference(c->docs[d], var_gamma[d], phi, lambda[d]);
    }
    if (w_asgn_file != NULL)
        fclose(w_asgn_file);
    
    if(ss)
        free_suffstats(ss);

    for (d = 0; d < c->num_docs; d++)
        if(var_gamma[d])
            delete [] var_gamma[d];
    if(var_gamma)
        delete [] var_gamma;

    for (n = 0; n < max_length; n++)
        if(phi[n])
            delete [] phi[n];
    if(phi)
        delete [] phi;

    for (int i = 0; i < num_docs; i ++)
        if(lambda[i])
            delete [] lambda[i];
    if(lambda)    
        delete [] lambda;
}
double slda::zeta_xi_estimation(suffstats * ss, int eta_update, int step, int sample_size)
{
    double zeta_sum, xi_sum;
    double ro = pow(step + delay, -forgetting_rate);
    double D = num_docs/double(sample_size);
	//printf("sample size: %d (D=%.3f)\n", sample_size, D);
    int k, w, a, c_, l;
    double sum_sqr = 0, rmse = -1;

    for (k = 0; k < num_topics; k++) {
        zeta_sum = 0;
        for (w = 0; w < size_vocab; w++)
        {
            zeta[k][w] = (1-ro)*zeta[k][w] + ro*(tau + D*ss->word_ss[k][w]);
            zeta_sum += zeta[k][w];
        }
        for (w = 0; w < size_vocab; w++)
            exp_log_beta[k][w] = digamma(zeta[k][w]) - digamma(zeta_sum);
    }

    if(pi_estimation && eta_update)
    {
        for(a = 0; a < num_annot; a++)
            for (c_ = 0; c_ < num_classes; c_++) 
            {
                xi_sum = 0;
                for (l = 0; l < num_classes; l++)
                { 
                    xi[a][c_][l] = (1-ro)*xi[a][c_][l] + ro*(omega + D*ss->pi_stat[a][c_][l]);
                    xi_sum += xi[a][c_][l];
                }
                for (l = 0; l < num_classes; l++)
                { 
                    exp_log_pi[a][c_][l] = digamma(xi[a][c_][l]) - digamma(xi_sum);
                    pi_est[a][c_][l] = xi[a][c_][l] / xi_sum;
                }
            }

        FILE * pi_fileptr = fopen(pi_filename, "r");
        if(pi_fileptr != NULL)
        {
            for(a = 0; a < num_annot; a++)
                for (c_ = 0; c_ < num_classes; c_++) 
                    for (l = 0; l < num_classes; l++) 
                        sum_sqr += (pi_est[a][c_][l] - true_pi[a][c_][l]) * (pi_est[a][c_][l] - true_pi[a][c_][l]);

            rmse = sqrt(sum_sqr/(num_annot*num_classes*num_classes));
            printf("Pi RMSE: %f\n", rmse);
            fclose(pi_fileptr);
        }
		else {
			printf("\n");
		}
    }
    else {
    	printf("\n");
    }

    return rmse;

}
void slda::mle(suffstats * ss, int eta_update, double ** lambda, corpus* c) // M-STEP
{

    int k, l;

    if (eta_update == 0) {
        return;
	}

    printf("maximizing ...\n");
	double f = 0.0;
	int status;
	int opt_iter;
	int opt_size = (num_classes-1) * num_topics;

	opt_parameter param;
	param.ss = ss;
	param.model = this;
	param.PENALTY = penalty;
    param.lambda = lambda;

	const gsl_multimin_fdfminimizer_type * T;
	gsl_multimin_fdfminimizer * s;
	gsl_vector * x;
	gsl_multimin_function_fdf opt_fun;
	opt_fun.f = &softmax_f;
	opt_fun.df = &softmax_df;
	opt_fun.fdf = &softmax_fdf;
	opt_fun.n = opt_size;
	opt_fun.params = (void*)(&param);
	x = gsl_vector_alloc(opt_size);

	for (l = 0; l < num_classes-1; l ++)
	{
		for (k = 0; k < num_topics; k ++)
		{
			gsl_vector_set(x, l*num_topics + k, eta[l][k]);
		}
	}

	T = gsl_multimin_fdfminimizer_vector_bfgs;
	s = gsl_multimin_fdfminimizer_alloc(T, opt_size);
	gsl_multimin_fdfminimizer_set(s, &opt_fun, x, 0.02, 1e-4);

	opt_iter = 0;
	do
	{
		opt_iter ++;
		status = gsl_multimin_fdfminimizer_iterate(s);
		if (status)
			break;
		status = gsl_multimin_test_gradient(s->gradient, 1e-3);
		if (status == GSL_SUCCESS)
			break;
		f = -s->f;
		if ((opt_iter-1) % 10 == 0)
			printf("step: %02d -> f: %f\n", opt_iter-1, f);
	} while (status == GSL_CONTINUE && opt_iter < MSTEP_MAX_ITER);

	for (l = 0; l < num_classes-1; l ++)
	{
		for (k = 0; k < num_topics; k ++)
        {
			eta[l][k] = gsl_vector_get(s->x, l*num_topics + k);
        }
    }

	gsl_multimin_fdfminimizer_free (s);
	gsl_vector_free (x);

	printf("final f: %f\n", f);
}


double slda::doc_e_step(document* doc, double* gamma, double** phi, double * lambda,
                        suffstats * ss, int eta_update) // E-STEP
{
    double likelihood = 0.0;

    if (eta_update == 1)
        likelihood = slda_inference(doc, gamma, phi, lambda);
    else
        likelihood = lda_inference(doc, gamma, phi);

    int d = ss->num_docs;
    int n, k, i, a, c, idx;

    for(a = 0; a < num_annot; a++)
        for(c = 0; c < num_classes; c++)
        {
            if(doc->answers[a]!=-1)
            {
                ss->pi_stat[a][c][doc->answers[a]] += lambda[c];
                ss->pi_total_stat[a][c] += lambda[c];

            }
        }

    for (n = 0; n < doc->length; n++)
    {
        for (k = 0; k < num_topics; k++)
        {
            ss->word_ss[k][doc->words[n]] += doc->counts[n]*phi[n][k];
            ss->word_total_ss[k] += doc->counts[n]*phi[n][k];

            //statistics for each document of the supervised part
            ss->z_bar[d].z_bar_m[k] += doc->counts[n] * phi[n][k]; //mean
           
            for (i = k; i < num_topics; i ++) //variance
            {
                idx = map_idx(k, i, num_topics);
               
                if (i == k)
                    ss->z_bar[d].z_bar_var[idx] +=
                        doc->counts[n] * doc->counts[n] * phi[n][k]; 

                ss->z_bar[d].z_bar_var[idx] -=
                    doc->counts[n] * doc->counts[n] * phi[n][k] * phi[n][i];
            }
        }
    }

    for (k = 0; k < num_topics; k++)
    {
        ss->z_bar[d].z_bar_m[k] /= (double)(doc->total);
    }

    for (i = 0; i < num_topics*(num_topics+1)/2; i ++)
    {
        ss->z_bar[d].z_bar_var[i] /= (double)(doc->total * doc->total);
    }

    ss->num_docs = ss->num_docs + 1; //because we need it for store statistics for each docs

    return (likelihood);
}

double slda::lda_inference(document* doc, double* var_gamma, double** phi)
{
    int k, n, var_iter;
    double converged = 1, phisum = 0, likelihood = 0, likelihood_old = 0;
    double *oldphi = new double [num_topics];
    double *digamma_gam = new double [num_topics];

    // compute posterior dirichlet
    for (k = 0; k < num_topics; k++)
    {
        // F: initilize gamma_k = alpha + N/K (Step 2 of Figure 6 of Blei2003)
        var_gamma[k] = alpha + (doc->total/((double) num_topics)); 
        digamma_gam[k] = digamma(var_gamma[k]);

        for (n = 0; n < doc->length; n++)
            // F: initilize phi_nk = 1/K (Step 1 of Figure 6 of Blei2003)
            phi[n][k] = 1.0/num_topics; 
    }

    var_iter = 0;
    while (converged > var_converged && (var_iter < var_max_iter || var_max_iter == -1))
    {
        var_iter++;
        for (n = 0; n < doc->length; n++) // STEP 4 FIGURE 6 OF LDA
        {
            phisum = 0;

            for (k = 0; k < num_topics; k++) // STEP 5 FIGURE 6 OF LDA
            {
                oldphi[k] = phi[n][k];

                // F: standard phi update from Figure 6, Step 6 of Blei2003, but IN LOG SPACE! hence the product becomes a sum...
                phi[n][k] = digamma_gam[k] + exp_log_beta[k][doc->words[n]]; 

                // F: this computes the normalization constant of phi; for example the digamma(sum_j phi_j) comes from here...
                if (k > 0)
                    phisum = log_sum(phisum, phi[n][k]); 
                else
                    phisum = phi[n][k]; // note, phi is in log space
            }

            // F: normalize the phis and update the gammas (Step 8 from Figure 6 of Blei2003)
            for (k = 0; k < num_topics; k++)
            {
                // F: normalize and move back from log space, by exponentiating...
                phi[n][k] = exp(phi[n][k] - phisum); // normalize

                // F: this update is in a sequencial form; to verify that it is correct notice the following:
                // gamma^(t+1) = alpha + sum_n phi_n^(t+1) = alpha + sum_n phi_n^(t+1) + sum_n phi_n^t - sum_n phi_n^t = gamma^t + sum(phi_n^(t+1) - phi_n^t)
                var_gamma[k] = var_gamma[k] + doc->counts[n]*(phi[n][k] - oldphi[k]);
                
                // !!! a lot of extra digamma's here because of how we're computing it
                // !!! but its more automatically updated too.
                digamma_gam[k] = digamma(var_gamma[k]);
            }
        }

        likelihood = lda_compute_likelihood(doc, phi, var_gamma);
        assert(!isnan(likelihood));
        converged = (likelihood_old - likelihood) / likelihood_old;
        likelihood_old = likelihood;
    }

    delete [] oldphi;
    delete [] digamma_gam;

    return likelihood;
}

double slda::lda_compute_likelihood(document* doc, double** phi, double* var_gamma)
{
    double likelihood = 0, digsum = 0, var_gamma_sum = 0;
    double *dig = new double [num_topics];
    int k, n;
    double alpha_sum = num_topics * alpha;
    
    for (k = 0; k < num_topics; k++)
    {
        dig[k] = digamma(var_gamma[k]);
        var_gamma_sum += var_gamma[k];
    }
    
    digsum = digamma(var_gamma_sum);

    likelihood = lgamma(alpha_sum) - lgamma(var_gamma_sum);


    for (k = 0; k < num_topics; k++)
    {
        likelihood += - lgamma(alpha) + (alpha - 1)*(dig[k] - digsum) +
                      lgamma(var_gamma[k]) - (var_gamma[k] - 1)*(dig[k] - digsum);

        for (n = 0; n < doc->length; n++)
        {
            if (phi[n][k] > 0)
            {
                likelihood += doc->counts[n]*(phi[n][k]*((dig[k] - digsum) -
                                              log(phi[n][k]) + exp_log_beta[k][doc->words[n]]));
            }
        }
    }

    delete [] dig;
    return likelihood;
}

double slda::zeta_xi_likelihood()
{
    double tau_sum = size_vocab * tau;
    double omega_sum = num_classes * omega;
    double zeta_sum, xi_sum, likelihood = 0;
    int a, c, l, k, v;

    likelihood = lgamma(tau_sum)*num_topics + lgamma(omega_sum)*num_classes*num_annot;
    for (a = 0; a < num_annot; a++)
    {
        for (c = 0; c < num_classes; c++)
        {
            xi_sum = 0;
            for(l = 0; l < num_classes; l++)
            {
                likelihood += -lgamma(omega) + (omega - 1)*(exp_log_pi[a][c][l]) - lgamma(xi[a][c][l]) + (xi[a][c][l] - 1)*(exp_log_pi[a][c][l]);
                xi_sum += xi[a][c][l];
            }
            likelihood -= lgamma(xi_sum);
        }
    }

    for (k = 0; k < num_topics; k++)
    {
        zeta_sum = 0;
        for(v = 0; v < size_vocab; v++)
        {
            likelihood += -lgamma(tau) + (tau - 1)*(exp_log_beta[k][v]) - lgamma(zeta[k][v]) + (zeta[k][v] - 1)*(exp_log_beta[k][v]); 
            zeta_sum += zeta[k][v];
        }
        likelihood -=  lgamma(zeta_sum);
    }

    return likelihood;
}


double slda::slda_compute_likelihood(document* doc, double** phi, double* var_gamma, double *lambda)
{
    double likelihood = 0, digsum = 0, var_gamma_sum = 0, t = 0.0, t1 = 0.0, t2 = 0.0;
    double * dig = new double [num_topics];
    int k, n, l, c, a;
    double alpha_sum = num_topics * alpha;
  
    for (k = 0; k < num_topics; k++)
    {
        dig[k] = digamma(var_gamma[k]);
        var_gamma_sum += var_gamma[k];
    }
  
    digsum = digamma(var_gamma_sum);

    likelihood = lgamma(alpha_sum) - lgamma(var_gamma_sum);
    t = 0.0;
    for (k = 0; k < num_topics; k++)
    {
        likelihood += -lgamma(alpha) + (alpha - 1)*(dig[k] - digsum) + lgamma(var_gamma[k]) - (var_gamma[k] - 1)*(dig[k] - digsum);

        for (n = 0; n < doc->length; n++)
        {
            if (phi[n][k] > 0)
            {
                likelihood += doc->counts[n]*(phi[n][k]*((dig[k] - digsum) - log(phi[n][k]) + exp_log_beta[k][doc->words[n]]));
               
                for(c = 0; c < num_classes-1; c++)
                    t += lambda[c] * eta[c][k] * doc->counts[n] * phi[n][k];
            }
        }
    }

    likelihood += t / (double)(doc->total); 	//eta_k*\bar{\phi}

    // h^T * \phi_n:
    t = 1.0; //the class model->num_classes-1
    for (l = 0; l < num_classes-1; l ++)
    {
        likelihood -= lambda[l] * log(lambda[l]); 

        for(a = 0; a<num_annot; a++)
            if(doc->answers[a]!=-1)
                likelihood += lambda[l] * exp_log_pi[a][l][doc->answers[a]]; 

        t1 = 1.0; 
        for (n = 0; n < doc->length; n ++)
        {
            t2 = 0.0;
            for (k = 0; k < num_topics; k ++)
            {
                t2 += phi[n][k] * exp(eta[l][k] * doc->counts[n]/(double)(doc->total));
            }
            t1 *= t2; 
        }
        t += t1;
    }

    likelihood -= log(t); 
    delete [] dig;

    return likelihood;
}

double slda::slda_inference(document* doc, double* var_gamma, double** phi, double* lambda)
{
    int k, n, var_iter, l;
    int FP_MAX_ITER = 10;
    int fp_iter = 0;
    double converged = 1, lambda_sum = 0, sl = 0, dot_prod = 0, phisum = 0, phisum_n, likelihood = 0, likelihood_old = 0;
    double * pp = new double [num_classes];
    double * phimean_n = new double [num_topics];
    double * oldphi = new double [num_topics];
    double * digamma_gam = new double [num_topics];
    double * sf_params = new double [num_topics];
    double * sf_aux = new double [num_classes-1];
    double sf_val = 0.0;

    // compute posterior dirichlet
    for (k = 0; k < num_topics; k++)
    {
        var_gamma[k] = alpha + (doc->total/((double) num_topics));
        digamma_gam[k] = digamma(var_gamma[k]);
        
        for (n = 0; n < doc->length; n++)
            phi[n][k] = 1.0/(double)(num_topics);
    }

    double t = 0.0;
    for (l = 0; l < num_classes-1; l ++)
    {
        sf_aux[l] = 1.0; // the quantity for equation 6 of each class
        // F: compute prod_n ( sum_j phi_nj * exp(eta_lj / N) )
        for (n = 0; n < doc->length; n ++)
        {
            t = 0.0; // F: t = sum_j phi_nj * exp(eta_lj / N)
            for (k = 0; k < num_topics; k ++)
            {
                t += phi[n][k] * exp(eta[l][k] * doc->counts[n]/(double)(doc->total));
            }
            sf_aux[l] *= t;
        }
    }

    var_iter = 0;

    while ((converged > var_converged) && ((var_iter < var_max_iter) || (var_max_iter == -1)))
    {
        var_iter++;
        for (n = 0; n < doc->length; n++)
        {
            //compute sf_params
            // F: this computes the h vector from Wang2009 (sf_params in the code), for the linear function h^T phi_n of, for instance, eq. 7;
            // the way it is computed, is through the value of eq. 6 (sf_aux in the code), by removing the value of phi_n through divion
            memset(sf_params, 0, sizeof(double)*num_topics); //in log space
            for (l = 0; l < num_classes-1; l ++)
            {
                t = 0.0;
                for (k = 0; k < num_topics; k ++)
                {
                    t += phi[n][k] * exp(eta[l][k] * doc->counts[n]/(double)(doc->total));
                }
                sf_aux[l] /= t; //take out word n
                // F: this takes out "sum_k phi_nk exp(eta_lk / N)" from sf_aux

                for (k = 0; k < num_topics; k ++)
                {
                    //h in the paper
                    // F: h (sf_params) is then computed by summing sf_aux over all classes l;
                    // the reason for the exp() appearing multiplied, is because we removed it when taking out the word n (the "for" loop above)
                    sf_params[k] += sf_aux[l]*exp(eta[l][k] * doc->counts[n]/(double)(doc->total));
                }
            }
            //
            for (k = 0; k < num_topics; k++)
            {
                oldphi[k] = phi[n][k];
            }
            // F: this inner loop is not present in standard LDA; it appear here because of the second lower bound that we establish on the log-likelihood;
            // the parameters of the second approximation are updated to maximize the bound by using a fixed-point iteration... hence this loop...
            for (fp_iter = 0; fp_iter < FP_MAX_ITER; fp_iter ++) //fixed point update
            {
                // F: sf_val is the dot product "h^T phi_n"; this value need to be updated every time we move from phi_n^(old) to a new phi_n,
                // hence, it appears inside the for loop of the fixed-point updates.
                sf_val = 1.0; // the base class, in log space
                for (k = 0; k < num_topics; k++)
                    sf_val += sf_params[k]*phi[n][k];

                phisum = 0;
                for (k = 0; k < num_topics; k++)
                {
                    // F: the next 3 statments compute eq. 8 of Wang2009, but in log space...
                    phi[n][k] = digamma_gam[k] + exp_log_beta[k][doc->words[n]];

                    //added softmax parts
                    sl = 0;
                    for(int c = 0; c < num_classes-1; c++) 
                        phi[n][k] += lambda[c]*eta[c][k]/(double)(doc->total);  
                       
                    phi[n][k] -= sf_params[k]/(sf_val*(double)(doc->counts[n]));

                    if (k > 0)
                        phisum = log_sum(phisum, phi[n][k]);
                    else
                        phisum = phi[n][k]; // note, phi is in log space
                }
                for (k = 0; k < num_topics; k++)
                {
                    phi[n][k] = exp(phi[n][k] - phisum); //normalize

                }
            }  

            //back to sf_aux value
            // F: update the sf_aux variable according to the newly estimated phi_nk; this will give rise to new h vector (sf_params)
            for (l = 0; l < num_classes-1; l ++)
            {
                t = 0.0;
                for (k = 0; k < num_topics; k ++)
                    t += phi[n][k] * exp(eta[l][k] * doc->counts[n]/(double)(doc->total));
               
                // F: while previously we took out "sum_k phi_nk exp(eta_lk / N)" from sf_aux, now we are putting it back, 
                // but with a new value for phi_nk in it
                sf_aux[l] *= t;
            }

            for (k = 0; k < num_topics; k++)
            {
                var_gamma[k] = var_gamma[k] + doc->counts[n]*(phi[n][k] - oldphi[k]);
                digamma_gam[k] = digamma(var_gamma[k]);
            }
        }

        for (k = 0; k < num_topics; k++)
        {
            phisum_n = 0;
            for (int w = 0; w < doc->length; w++)
                phisum_n += phi[w][k] * doc->counts[w];

            phimean_n[k] = phisum_n /(double)(doc->total);
        }

        lambda_sum = 0;
        for (l = 0; l < num_classes-1; l ++)
        {
            dot_prod = 0;
            for(k = 0; k < num_topics; k ++)
                dot_prod += eta[l][k] * phimean_n[k];

            pp[l] = 0;
            for(int a = 0; a < num_annot; a++)
                if(doc->answers[a]!=-1)
                    pp[l] += exp_log_pi[a][l][doc->answers[a]];

            lambda[l] = exp(dot_prod + pp[l]) + lambda_smoother; 
            lambda_sum+=lambda[l]; 
        }

        pp[num_classes-1] = 0;
        for(int a = 0; a < num_annot; a++)
            if(doc->answers[a]!=-1)
                pp[num_classes-1] += exp_log_pi[a][num_classes-1][doc->answers[a]];
        
        lambda[num_classes-1] = exp(pp[num_classes-1]) + lambda_smoother;
        lambda_sum+=lambda[num_classes-1];

        for (l = 0; l < num_classes; l ++)
            lambda[l] /= lambda_sum;

        likelihood = slda_compute_likelihood(doc, phi, var_gamma, lambda);
        assert(!isnan(likelihood));
        converged = fabs((likelihood_old - likelihood) / likelihood_old);
        likelihood_old = likelihood;
    }

    delete [] pp;
    delete [] phimean_n; 
    delete [] oldphi;
    delete [] digamma_gam;
    delete [] sf_params;
    delete [] sf_aux;
    return likelihood;
}

void slda::infer_only(corpus * c, const char * directory)
{
    int i, k, d, n, c_;
    double **var_gamma, likelihood, **phi;
    double* phi_m;
    char filename[1000];
    double base_score, score;
    int label;
    int num_correct = 0;
    int max_length = c->max_corpus_length();
    int * accuracy_count = new int [num_classes];
    memset(accuracy_count, 0, sizeof(int)*num_classes); 

    docs_by_class = new int[num_classes];
    memset(docs_by_class, 0, sizeof(int)*num_classes);

    var_gamma = new double * [c->num_docs];
    for (i = 0; i < c->num_docs; i++)
        var_gamma[i] = new double [num_topics];

    phi = new double * [max_length];
    for (n = 0; n < max_length; n++)
        phi[n] = new double [num_topics];

    phi_m = new double [num_topics];

    FILE * likelihood_file = NULL;
    sprintf(filename, "%s/inf-likelihood.dat", directory);
    likelihood_file = fopen(filename, "w");
    FILE * inf_label_file = NULL;
    sprintf(filename, "%s/inf-labels.dat", directory);
    inf_label_file = fopen(filename, "w");

    for (d = 0; d < c->num_docs; d++)
    {
        if ((d % 1000) == 0)
            printf("document %d\n", d);

        document * doc = c->docs[d];

        docs_by_class[doc->label]++;

        likelihood = lda_inference(doc, var_gamma[d], phi);

        memset(phi_m, 0, sizeof(double)*num_topics); //zero_initialize
        for (n = 0; n < doc->length; n++)
            for (k = 0; k < num_topics; k ++)
                phi_m[k] += doc->counts[n] * phi[n][k];

        for (k = 0; k < num_topics; k ++)
            phi_m[k] /= (double)(doc->total);
        
        //do classification
        label = num_classes-1;
        base_score = 0.0;
        for (i = 0; i < num_classes-1; i ++)
        {
            score = 0.0; 
            
            for (k = 0; k < num_topics; k ++)
                score += eta[i][k] * phi_m[k];

            if (score > base_score)
            {
                base_score = score;
                label = i;
            }
        }

        if (label == doc->label)
        {
            accuracy_count[doc->label]++;
            num_correct++;
        }

        fprintf(likelihood_file, "%5.5f\n", likelihood);
        fprintf(inf_label_file, "%d\n", label);
    }

    printf("Accuracies by class:\n");
    for(c_ = 0; c_< num_classes; c_++)
        printf("[%d] %.3f\n", c_, (double)accuracy_count[c_] / (double) docs_by_class[c_]);
    printf("Average accuracy: %.3f\n", (double)num_correct / (double) c->num_docs);

    sprintf(filename, "%s/inf-gamma.dat", directory);
    save_gamma(filename, var_gamma, c->num_docs);

    for (d = 0; d < c->num_docs; d++)
        delete [] var_gamma[d];
    delete [] var_gamma;
    for (n = 0; n < max_length; n++)
        delete [] phi[n];
    delete [] phi;
    delete [] phi_m;
}

void slda::save_gamma(char* filename, double** gamma, int num_docs)
{
    int d, k;

    FILE* fileptr = fopen(filename, "w");
    if (fileptr != NULL) {
        for (d = 0; d < num_docs; d++)
        {
            fprintf(fileptr, "%5.10f", gamma[d][0]);
            for (k = 1; k < num_topics; k++)
                fprintf(fileptr, " %5.10f", gamma[d][k]);
            fprintf(fileptr, "\n");
        }
        fclose(fileptr);
    }
}

void slda::write_word_assignment(FILE* f, document* doc, double** phi)
{
    int n;
    fprintf(f, "%03d", doc->length);

    for (n = 0; n < doc->length; n++)
        if(doc->words[n] && phi[n])
        {
            fprintf(f, " %04d:%02d", doc->words[n], argmax(phi[n], num_topics));
        }
   
    fprintf(f, "\n");
    fflush(f);
}
