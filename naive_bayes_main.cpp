#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <chrono>

#include <algorithm>
#include <random>
#include <iterator>
#include <mutex>

#include "thread_provider.h"

using namespace std;
using namespace std::chrono;

class cross_validation_result
{
public:
    float avg_accuracy, min_accuracy, max_accuracy, accuracy_variance;
    cross_validation_result()
    {
        avg_accuracy = accuracy_variance = 0;
        min_accuracy = std::numeric_limits<float>::max();
        max_accuracy = std::numeric_limits<float>::min();
    }
};

class naive_bayes_classifier
{
private:
    vector<float> spam_mean,
        ham_mean,
        spam_variance,
        ham_variance;
    float spam_probability,
        ham_probability;

public:
    naive_bayes_classifier() = default;
    ~naive_bayes_classifier() = default;

    float score(vector<vector<float>> &dataset)
    {
        int dataset_size = dataset.size();
        float prob_of_x_given_spam, prob_of_x_given_ham;
        float total = 0.;
        for (int j = 0; j < dataset_size; j++)
        {
            prob_of_x_given_ham = this->ham_probability;
            prob_of_x_given_spam = this->spam_probability;
            for (int i = 0; i < 54; i++)
            {
                prob_of_x_given_spam *= pow((2 * M_PI * spam_variance[i]), (-0.5)) *
                                        exp((dataset[j][i] - spam_mean[i]) * (dataset[j][i] - spam_mean[i]) / (-2 * spam_variance[i]));
                prob_of_x_given_ham *= pow((2 * M_PI * ham_variance[i]), (-0.5)) *
                                       exp((dataset[j][i] - ham_mean[i]) * (dataset[j][i] - ham_mean[i]) / (-2 * ham_variance[i]));
            }
            total += ((prob_of_x_given_spam > prob_of_x_given_ham) == dataset[j][57]);
        }
        cout << "TOTAL IN SCORE: --- " << total << std::endl;
        cout << "TOTAL/MEAN IN SCORE: --- " << total / dataset_size << std::endl;
        return total / dataset_size;
    }

    void fit(vector<vector<float>> &dataset)
    {
        // TODO: Split Ham and Spam emails in Cross Validation to pass them already split into fit and score
        int spam_frequency, ham_frequency;
        float size = static_cast<float>(dataset.size());
        spam_frequency = ham_frequency = 0;
        vector<float> spam_totals(54, 0.);
        vector<float> ham_totals(54, 0.);
        spam_mean = vector<float>(54, 0.);
        ham_mean = vector<float>(54, 0.);
        spam_variance = vector<float>(54, 0.);
        ham_variance = vector<float>(54, 0.);
        int i;
        // Mean pre computing
        for (vector<float> &row : dataset)
        {
            if (row[57]) // Label == 1 -> is spam
            {
                spam_frequency++;
                for (i = 0; i < 54; i++)
                {
                    spam_totals[i] += row[i];
                }
            }
            else
            {
                ham_frequency++;
                for (i = 0; i < 54; i++)
                {
                    ham_totals[i] += row[i];
                }
            }
        }
        // cout << "PROPORTION OF SPAM IN FIT: --- " << spam_frequency / size << std::endl;
        // cout << "PROPORTION OF HAM IN FIT: --- " << ham_frequency / size << std::endl;
        this->spam_probability = spam_frequency / size;
        this->ham_probability = ham_frequency / size;
        // Mean Computation
        float spam_size, ham_size;
        for (i = 0; i < 54; i++)
        {
            this->spam_mean[i] = spam_totals[i] / spam_frequency;
            this->ham_mean[i] = ham_totals[i] / ham_frequency;
        }
        // Variance pre computing: here spam totals will be used to cumulatively store the sum of variances
        std::fill(spam_totals.begin(), spam_totals.end(), 0);
        std::fill(ham_totals.begin(), ham_totals.end(), 0);
        for (int j = 0; j < spam_frequency; j++)
        {
            for (i = 0; i < 54; i++)
            {
                spam_totals[i] += ((dataset[j][i] - this->spam_mean[i]) * (dataset[j][i] - this->spam_mean[i]));
            }
        }
        for (int j = 0; j < ham_frequency; j++)
        {
            for (i = 0; i < 54; i++)
            {
                ham_totals[i] += ((dataset[j][i] - this->ham_mean[i]) * (dataset[j][i] - this->ham_mean[i]));
            }
        }
        // Variance computation
        float spamTotal, hamTotal;
        spamTotal = hamTotal = 0.;
        for (i = 0; i < 54; i++)
        {
            this->spam_variance[i] = spam_totals[i] / spam_frequency + std::numeric_limits<float>::min();
            spamTotal += this->spam_variance[i];
            this->ham_variance[i] = ham_totals[i] / ham_frequency + std::numeric_limits<float>::min();
            hamTotal += this->ham_variance[i];
        }
        // cout << "TOTAL VARIANCE FOR SPAM AND HAM: " << spamTotal << "   " << hamTotal << endl;
    }
};

void print_dataset(vector<vector<float>> &dataset)
{
    for (vector<float> &row : dataset)
    {
        for (float &value : row)
        {
            cout << value << ", ";
        }
        cout << "\n";
    }
}

vector<vector<float>> read_dataset(string filePath)
{
    ifstream indata;
    indata.open(filePath);
    if (!indata)
    {
        cerr << "Error: file could not be opened" << endl;
        exit(1);
    }
    vector<vector<float>> allData;
    string line;
    while (getline(indata, line))
    {
        // cout << "Line: " << line << "\n";
        stringstream ss(line);
        vector<float> row;
        string substr;
        while (getline(ss, substr, ','))
        {
            // cout << "Value: " << substr << " ";
            row.push_back(stof(substr));
        }
        // cout << "\n";
        allData.push_back(row);
    }
    return allData;
}

/**
 * @brief The following function performs a ten fold cross-validation on the given dataset, first fitting
 * through Naive Bayes, then firing a thread for each fold configuration to compute the testing score.
 *
 * @param mails_dataset
 */
cross_validation_result __10_folds_cross_validation(vector<vector<float>> &mails_dataset)
{

    // Shuffles the dataset
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(mails_dataset.begin(), mails_dataset.end(), g);
    // Variables init
    int dataset_size = mails_dataset.size(),
        fold_size = dataset_size / 10,
        test_fold_start,
        test_fold_end,
        other_folds_start;
    float total_score, minimum_score, avarage_score, maximum_score, score_variance;
    avarage_score = score_variance = total_score = 0;
    minimum_score = std::numeric_limits<float>::max();
    maximum_score = std::numeric_limits<float>::min();
    // vector<vector<float>> test_fold, other_folds;
    vector<vector<float>>::const_iterator dataset_begin = mails_dataset.begin();
    // Thread pool initialization
    thread_provider<mutex> pool(25);
    mutex total_score_mutex;

    vector<float> all_scores(10);
    cross_validation_result cv_result;

    // CROSS VALIDATION.
    for (uint8_t validation_iteration = 0; validation_iteration < 10; validation_iteration++)
    {
        pool.executeTask(
            [&, validation_iteration, dataset_begin]()
            {
                vector<vector<float>> test_fold, other_folds;
                test_fold_start = validation_iteration * fold_size;
                test_fold_end = (validation_iteration == 9 &&
                                 (dataset_size % 10) != 0)
                                    ? dataset_size - 1
                                    : test_fold_start + fold_size - 1;
                test_fold = vector<vector<float>>(dataset_begin + test_fold_start,
                                                  dataset_begin + test_fold_end);
                other_folds = vector<vector<float>>(mails_dataset);
                other_folds.erase(other_folds.begin() + test_fold_start, other_folds.begin() + test_fold_end);

                // FIT AND SCORE
                naive_bayes_classifier classifier;
                classifier.fit(other_folds);
                {
                    const std::lock_guard<std::mutex> lock(total_score_mutex);
                    all_scores[validation_iteration] = classifier.score(test_fold);
                    total_score += all_scores[validation_iteration];
                }
            });
    }
    pool.shutdown();
    cv_result.avg_accuracy = total_score / 10.;
    for (float &single_result : all_scores)
    {
        cv_result.min_accuracy = (single_result < cv_result.min_accuracy) ? single_result : cv_result.min_accuracy;
        cv_result.max_accuracy = (single_result > cv_result.max_accuracy) ? single_result : cv_result.max_accuracy;
        cv_result.accuracy_variance += (single_result - cv_result.avg_accuracy) * (single_result - cv_result.avg_accuracy);
    }
    return cv_result;
}

/**
 * @brief The following function takes the whole dataset and splits it in two vectors, respectively containing
 * the first 54 cells (the input features) of each mail and the class label.
 *
 * @param mails_dataset  Represents the starting dataset.
 * @param X Is the vector of features for each mail.
 * @param y Is the vector of class labels corresponding to each mail.
 */
void init_trainig_dataset(vector<vector<float>> &mails_dataset, vector<vector<float>> &X, vector<float> &y)
{
    for (vector<float> &row : mails_dataset)
    {
        X.push_back(vector<float>(row.begin(), row.begin() + 54));
        // cout << "SIZE: " << X[0].size() << std::endl;
        y.push_back(row[57]);
    }
}

int main(int argc, char const *argv[])
{
    vector<vector<float>> mails_dataset = read_dataset("spambase.data");
    steady_clock::time_point begin = steady_clock::now();
    cross_validation_result result = __10_folds_cross_validation(mails_dataset);
    steady_clock::time_point end = steady_clock::now();
    double elapsedTime = static_cast<double>(duration_cast<microseconds>(end - begin).count()) / 1000000;
    cout << "Elapsed time = " << elapsedTime << " seconds.\n"
         << endl
         << "MIN: " << result.min_accuracy << endl
         << "MEAN: " << result.avg_accuracy << endl
         << "MAX: " << result.max_accuracy << endl
         << "VARIANCE: " << result.accuracy_variance << endl
         << "STANDARD DEVIANCE: " << sqrt(result.accuracy_variance) << endl;
    return 0;
}
