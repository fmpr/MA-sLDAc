
************************************************************************************
MULTI-ANNOTATOR SUPERVISED LATENT DIRICHLET ALLOCATION FOR CLASSIFICATION (MA-sLDAc)
************************************************************************************

(C) Copyright 2015, Filipe Rodrigues (fmpr@dei.uc.pt), Mariana Lourenço, Bernardete Ribeiro and Francisco Pereira

MA-sLDA is an extension of supervised LDA (sLDA) (Wang et. al., 2009) to multiple-annotator settings. MA-sLDA is then capable of jointly modeling the words in documents as arising from a mixture of topics, as well as the latent true labels and the (noisy) labels of the multiple annotators. Part of the code is from http://www.cs.cmu.edu/~chongw/slda/

This is a C++ implementation of MA-sLDA for classification with batch variational inference. For further details please see the original paper: “Learning supervised topic models from crowds. Rodrigues, F. and Camara, F. and Ribeiro, B. The Third AAAI Conference on Human Computation and Crowdsourcing (HCOMP), 2015.”

maslda is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free
Software Foundation; either version 2 of the License, or (at your
option) any later version.


***** COMPILING *****

Type "make" in a shell. 
Please note that this code requires the Gnu Scientific Library, http://www.gnu.org/software/gsl/


***** ESTIMATION *****

USAGE: maslda est [data] [answers] [settings] [alpha] [tau] [omega] [k] [random/seeded/model_path] [seed]

EXAMPLE: ./maslda est ../20newsgroups/data_train.txt ../20newsgroups/answers.txt settings.txt 0.1 0.5 1 5 random 1 output

Data format:
[data] is a file where each line is of the form: [M] [term_1]:[count] [term_2]:[count] ...  [term_N]:[count], where [M] is the number of unique terms in the document, and the [count] associated with each term is how many times that term appeared in the document. 

[answers] is a file where each line contains the labels of the different annotators (separated by a white space) for [data]. Each column therefore corresponds to all the answers of an annotator. The labels must be 0, 1, ..., C-1, if we have C classes. Missing answers are encoded using “-1”.


***** INFERENCE *****

USAGE: maslda inf [data] [labels] [settings] [model] [directory]

EXAMPLE: ./maslda inf ../20newsgroups/data_test.txt ../20newsgroups/labels_test.txt settings.txt output/final.model output

Data format:
[labels] is a file where each line is the corresponding true label for [data].


***** SETTINGS *****

- “labels file” is a file with the true labels for the training documents. If a valid file is provided, it will be use to compute and report error statistics during the model estimation.
- “pi estimate” or “pi fixed” determines whether of not the confusion matrices of the different should be estimated or kept fixed to the values provided by “pi file”.
- “pi file” is a file with the true confusion matrices of the multiple annotators. If a valid file is provided, it will be use to compute and report error statistics during the model estimation.
- “lambda laplace smoother” defines the value of the laplace smoother used when estimating lambda.


