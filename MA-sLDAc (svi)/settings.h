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
#ifndef SETTINGS_H
#define SETTINGS_H
#include <stdio.h>
#include <string.h>
#include <stdlib.h>


struct settings
{
    float  VAR_CONVERGED;
    int    VAR_MAX_ITER;
    float  EM_CONVERGED;
    int    EM_MAX_ITER;
    int    ESTIMATE_ALPHA;
    float  PENALTY;
    int    ESTIMATE_PI;
    char   PI_FILE[100];
    char   LABELS_FILE[100];
    float  LAMBDA_SMOOTHER;
    float  FORGETTING_RATE;
    int    DELAY;

    int read_settings(char* filename)
    {
        FILE * fileptr;
        char alpha_action[100];
        char pi_action[100];
        char pi_file[100];
        char labels_file[100];

        fileptr = fopen(filename, "r");
        fscanf(fileptr, "var max iter %d\n", &this->VAR_MAX_ITER);
        fscanf(fileptr, "var convergence %f\n", &this->VAR_CONVERGED);
        fscanf(fileptr, "em max iter %d\n", &this->EM_MAX_ITER);
        fscanf(fileptr, "em convergence %f\n", &this->EM_CONVERGED);
        fscanf(fileptr, "L2 penalty %f\n", &this->PENALTY);

        fscanf(fileptr, "labels file %s\n", labels_file);
        memcpy(this->LABELS_FILE, labels_file, sizeof labels_file);
        printf("labels file %s\n", this->LABELS_FILE);

        fscanf(fileptr, "pi %s\n", pi_action);
        if (strcmp(pi_action, "fixed") == 0)
        {
            this->ESTIMATE_PI = 0;
            printf("pi is fixed ...\n");
        }
        else
        {
            this->ESTIMATE_PI = 1;
            printf("pi is esimated ...\n");
        }

        fscanf(fileptr, "pi file %s\n", pi_file);

        if(this->ESTIMATE_PI == 0 && (fopen(pi_file, "r") == NULL) )
        {
            printf("Error: if pi is fixed, a valid pi file has to be provided\n");
            return 0;
        }

        memcpy(this->PI_FILE, pi_file, sizeof pi_file);
        printf("pi file %s\n", this->PI_FILE);

        fscanf(fileptr, "lambda laplace smoother %f\n", &this->LAMBDA_SMOOTHER);
        fscanf(fileptr, "svi delay %d\n", &this->DELAY);
        fscanf(fileptr, "svi forgetting rate %f", &this->FORGETTING_RATE);
        fclose(fileptr);

        printf("var max iter %d\n", this->VAR_MAX_ITER);
        printf("var convergence %.2E\n", this->VAR_CONVERGED);
        printf("em max iter %d\n", this->EM_MAX_ITER);
        printf("em convergence %.2E\n", this->EM_CONVERGED);
        printf("L2 penalty %.2E\n", this->PENALTY);
        printf("Lambda smoother %f\n", this->LAMBDA_SMOOTHER);
        printf("Delay %d\n", this->DELAY);
        printf("Forgetting rate %f\n", this->FORGETTING_RATE);

        return 1;

    }

};

#endif // SETTINGS_H

