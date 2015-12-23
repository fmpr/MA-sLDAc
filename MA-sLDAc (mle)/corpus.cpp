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

#include "corpus.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>


corpus::corpus()
{
    num_docs = 0;
    size_vocab = 0;
    num_classes = 0;
    num_total_words = 0;
    num_annot = 0;
}

corpus::~corpus()
{
    for (int i = 0; i < num_docs; i ++)
    {
        document * doc = docs[i];
        delete doc;
    }
    docs.clear();

    num_docs = 0;
    size_vocab = 0;
    num_classes = 0;
    num_total_words = 0;



}

void corpus::read_data(const char * data_filename,
                       const char * label_filename,
                       const char * answers_filename) 
{
    int OFFSET = 0;
    int length = 0, value, word = 0, i,
        n = 0, nd = 0, nw = 0, label = -1, na = 0, first = 1;

    double count = 0;
    char * buffer = new char[90000];
    char *pbuff;

    FILE * fileptr;
    fileptr = fopen(data_filename, "r");
    printf("\nreading data from %s\n", data_filename);
    nd = 0;
    nw = 0;

    while ((fscanf(fileptr, "%10d", &length) != EOF))
    {
        document * doc = new document(length);

        for (n = 0; n < length; n++)
        {
            fscanf(fileptr, "%10d:%lf", &word, &count);
            word = word - OFFSET;
            doc->words[n] = word;
            doc->counts[n] = count; 
            doc->label = -1;

            doc->total += count;

            if (word >= nw)
            {
                nw = word + 1;
            }
        }
        num_total_words += doc->total;
        docs.push_back(doc);
        nd++;
    }
    fclose(fileptr);
    num_docs = nd;
    size_vocab = nw;
    printf("number of docs  : %d\n", nd);
    printf("number of terms : %d\n", nw);
    printf("number of total words : %d\n\n", num_total_words);

    fileptr = fopen(label_filename, "r");
    if (fileptr != NULL) 
    {
        printf("reading labels from %s\n", label_filename);
        nd = 0;
        while ((fscanf(fileptr, "%10d", &label) != EOF))
        {
            document * doc = docs[nd];
            doc->label = label;
            if (label >= num_classes)
            {
                num_classes = label + 1;
            }
            nd ++;
        }
        assert(nd == int(docs.size()));
    }

    if (answers_filename != NULL) 
    {
        fileptr = fopen(answers_filename, "r");
        printf("reading annotators answers from %s\n", answers_filename);
        for(nd = 0; nd < int(docs.size()); nd++) 
        {
            if (!fgets(buffer, sizeof(char)*90000, fileptr)) 
                break;
        
            pbuff = buffer;

            na = 0;

            while (first || na<num_annot) 
            {   

                if (*pbuff == '\n')
                    break;

                value = strtol(pbuff, &pbuff, 10);

                if (docs[0]->label == -1 && value >= num_classes)
                {
                    num_classes = value + 1;
                }

                docs[nd]->answers.push_back(value);

                na++;
               
            }

            if(first)
            {
                num_annot = na;
                first = 0;

                delete [] buffer; 

                buffer = new char[num_annot*int(docs.size())];

            }
            else
            {
                assert(na == num_annot);
            }

        }
        printf("number of classes : %d\n", num_classes);
    }
    

    delete [] buffer; 

    assert(nd == int(docs.size()));

    printf("number of annotators : %d\n\n", num_annot);

    pi = new double ** [num_annot];
    for(int c = 0; c < num_annot; c++)
    {   
        pi[c] = new double * [num_classes]; 
        for (int i = 0; i < num_classes; i ++)
        {
            pi[c][i] = new double [num_classes];
        }
    }


}

int corpus::max_corpus_length() 
{
    int max_length = 0;

    for (int d = 0; d < num_docs; d++) 
    {
        if (docs[d]->length > max_length)
            max_length = docs[d]->length;
    }
    return max_length;
}
