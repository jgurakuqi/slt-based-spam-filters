# slt-based-spam-filters


## REQUIREMENTS:

Write a spam filter using discrimitative and generative classifiers. Use the Spambase dataset which already 
represents spam/ham messages through a bag-of-words representations through a dictionary of 48 highly 
discriminative words and 6 characters. The first 54 features correspond to word/symbols frequencies; ignore 
features 55-57; feature 58 is the class label (1 spam/0 ham).

Perform SVM classification using linear, polynomial of degree 2, and RBF kernels over the TF/IDF representation.
Can you transform the kernels to make use of angular information only (i.e., no length)? Are they still positive 
definite kernels? Classify the same data also through a Naive Bayes classifier for continuous inputs, modelling 
each feature with a Gaussian distribution, resulting in the following model:

where αk is the frequency of class k, and μki, σ2ki are the means and variances of feature i given that the data 
is in class k.
Perform k-NN clasification with k=5
Provide the code, the models on the training set, and the respective performances in 10 way cross validation.

Explain the differences between the three models.



P.S. you can use a library implementation for SVM, but do implement the Naive Bayes on your own. As for k-NN, 
you can use libraries if you want, but it might just be easier to do it on your own.


## OUR NOTES:

The jupyter notebook file is the file used to display the results from all the algorithms applied to
the spam/ham dataset.

For the Naive Bayes it was required to implement it without making use of ML libraries, so we opted for
a C++ implementation to gain extra performance.


Before of running the jupyter notebook file named "main.ipynb", if the naive_bayes_main.o, lib_naive_.so and 
naivelib.so files are misssing, there is need to compile the c++ file with the following commands, in order
to export it as a callable library for Python:

    g++ -c -fPIC naive_bayes_main.cpp -O3 -Wextra -Wall -o naive_bayes_main.o
    g++ -shared -Wl,-soname,lib_naive.so -O3 -o lib_naive_.so  naive_bayes_main.o


If the user wishes to run the naive_bayes_main.cpp file directly, than it's advised to compile it with the 
following command:

    g++ naive_bayes_main.cpp -O3 -o out

and then run it through the following command (if you whish to run it from a bash terminal):

    ./out


## LICENSE


MIT License

Copyright (c) 2022 Jurgen Gurakuqi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
