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

#include <stdio.h>
#include <string.h>
#include "corpus.h"
#include "utils.h"
#include "maslda.h"

void help( void ) 
{
    printf("usage: maslda [est] [data] [answers] [settings] [alpha] [tau] [omega] [k] [random/seeded/model_path] [seed] [directory]\n");
    printf("       maslda [inf] [data] [labels] [settings] [model] [directory]\n");
}

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        help();
        return 0;
    }
    if (strcmp(argv[1], "est") == 0)
    {
        settings setting;
        char * setting_filename = argv[4];
        printf("setting_filename %s\n", setting_filename);

        if(!setting.read_settings(setting_filename))
            return 0;

        corpus c;
        char * data_filename = argv[2];
        char * answers_filename = argv[3];

        c.read_data(data_filename, setting.LABELS_FILE, answers_filename);
        
        double alpha = atof(argv[5]);
        printf("alpha %lf\n", alpha);
        double tau = atof(argv[6]);
        printf("tau %lf\n", tau);
        double omega = atof(argv[7]);
        printf("omega %lf\n", omega);
        int num_topics = atoi(argv[8]);
        printf("number of topics is %d\n", num_topics);
        char * init_method = argv[9];
        double init_seed = atof(argv[10]);
        char * directory = argv[11];
        printf("models will be saved in %s\n", directory);
        make_directory(directory);

        slda model;
        model.init(alpha, tau, omega, num_topics, &c, &setting, init_seed);
        model.v_em(&c, init_method, directory, &setting);

    }
    if (strcmp(argv[1], "inf") == 0)
    {
        corpus c;
        char * data_filename = argv[2];
        char * label_filename = argv[3];

        c.read_data(data_filename, label_filename, NULL);
        settings setting;
        char * setting_filename = argv[4];
        setting.read_settings(setting_filename);

        char * model_filename = argv[5];
        char * directory = argv[6];
        printf("\nresults will be saved in %s\n", directory);
        make_directory(directory);

        slda model;
        model.load_model(model_filename, &setting);
        model.infer_only(&c, directory);
    }

    return 0;
}
