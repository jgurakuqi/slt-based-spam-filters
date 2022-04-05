# spam_filter

The jupyter notebook file is the files used to display the results from all the algorithms applied to
the spam/ham dataset.

For the Naive Bayes it was required to implement it without making use of ML libraries, so we opted for
a C++ implementation to gain extra performance.


Before of running the jupyter notebook file named "main.ipynb", if the naive_bayes_main.o, naive_ba and 
naivelib.so files are misssing, there is need to compile the c++ file with the following commands, in order
to export it as a callable library for Python:


g++ -c -fPIC naive_bayes_main.cpp -O3 -Wextra -Wall -o naive_bayes_main.o
g++ -shared -Wl,-soname,lib_naive.so -O3 -o lib_naive_.so  naive_bayes_main.o