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

constexpr float ABSOLUTE_MIN = numeric_limits<float>::min();
constexpr float ABSOLUTE_MAX = numeric_limits<float>::max();

class cross_validation_result
{
public:
    float avg_accuracy, min_accuracy, max_accuracy, accuracy_variance;
    cross_validation_result()
    {
        avg_accuracy = accuracy_variance = 0;
        min_accuracy = ABSOLUTE_MAX;
        max_accuracy = ABSOLUTE_MIN;
    }

    void print_scores()
    {
        cout << endl
             << "Minimum Accuracy: " << this->min_accuracy << endl
             << "Average Accuracy: " << this->avg_accuracy << endl
             << "Maximum Accuracy: " << this->max_accuracy << endl
             << "Variance of Accuracy: " << this->accuracy_variance << endl
             << "Standard Deviation of Accuracy: " << sqrt(this->accuracy_variance) << endl;
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

    float score(vector<vector<float>> &test_set)
    {
        int test_set_size = test_set.size();
        float prob_of_x_given_spam, prob_of_x_given_ham;
        float total = 0.;
        for (int i = 0; i < test_set_size; i++)
        {
            prob_of_x_given_ham = this->ham_probability;
            prob_of_x_given_spam = this->spam_probability;
            for (int j = 0; j < 54; j++)
            {
                prob_of_x_given_spam *= pow((2 * M_PI * spam_variance[j]), (-0.5)) *
                                        exp((test_set[i][j] - spam_mean[j]) * (test_set[i][j] - spam_mean[j]) / (-2 * spam_variance[j]));
                prob_of_x_given_ham *= pow((2 * M_PI * ham_variance[j]), (-0.5)) *
                                       exp((test_set[i][j] - ham_mean[j]) * (test_set[i][j] - ham_mean[j]) / (-2 * ham_variance[j]));
            }
            total += ((prob_of_x_given_spam > prob_of_x_given_ham) == test_set[i][57]);
        }
        return total / test_set_size;
    }

    void fit(vector<vector<float>> &dataset)
    {
        int spam_frequency, ham_frequency;
        float dataset_size = static_cast<float>(dataset.size());
        spam_frequency = ham_frequency = 0;
        vector<float> spam_totals(54, 0.);
        vector<float> ham_totals(54, 0.);
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
        this->spam_probability = spam_frequency / dataset_size;
        this->ham_probability = ham_frequency / dataset_size;
        // Mean Computation
        float spam_size, ham_size;
        spam_mean = vector<float>(54, 0.);
        ham_mean = vector<float>(54, 0.);
        for (i = 0; i < 54; i++)
        {
            this->spam_mean[i] = spam_totals[i] / spam_frequency;
            this->ham_mean[i] = ham_totals[i] / ham_frequency;
            spam_totals[i] = 0;
            ham_totals[i] = 0;
        }
        // Variance pre computing: here spam totals will be used to cumulatively store the sum of variances
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
        spam_variance = vector<float>(54, 0.);
        ham_variance = vector<float>(54, 0.);
        for (i = 0; i < 54; i++)
        {
            this->spam_variance[i] = spam_totals[i] / spam_frequency + ABSOLUTE_MIN;
            this->ham_variance[i] = ham_totals[i] / ham_frequency + ABSOLUTE_MIN;
        }
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
            row.push_back(stof(substr));
        }
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
    // Variables init
    int dataset_size = mails_dataset.size(),
        fold_size = dataset_size / 10,
        test_fold_start,
        test_fold_end;
    float total_score = 0.;
    int irregular_folds = (dataset_size % 10) != 0;
    // vector<vector<float>> test_fold, train_folds;
    vector<vector<float>>::const_iterator dataset_begin = mails_dataset.begin();
    // Thread pool initialization

    vector<float> all_scores(10);

    thread_provider<mutex> pool;
    // CROSS VALIDATION.
    for (uint8_t validation_iteration = 0; validation_iteration < 10; validation_iteration++)
    {
        pool.executeTask(
            [&, validation_iteration, dataset_begin]()
            {
                vector<vector<float>> test_fold, train_folds;
                test_fold_start = validation_iteration * fold_size;
                test_fold_end = (validation_iteration == 9 &&
                                 irregular_folds)
                                    ? dataset_size - 1
                                    : test_fold_start + fold_size - 1;
                test_fold = vector<vector<float>>(dataset_begin + test_fold_start,
                                                  dataset_begin + test_fold_end);
                train_folds = vector<vector<float>>(mails_dataset);
                train_folds.erase(train_folds.begin() + test_fold_start, train_folds.begin() + test_fold_end);
                naive_bayes_classifier classifier;
                classifier.fit(train_folds);
                all_scores[validation_iteration] = classifier.score(test_fold);
            });
    }
    pool.shutdown();
    for (float &val : all_scores)
    {
        total_score += val;
    }
    cross_validation_result cv_result;
    cv_result.avg_accuracy = total_score / 10.;
    for (float &single_result : all_scores)
    {
        cv_result.min_accuracy = (single_result < cv_result.min_accuracy) ? single_result : cv_result.min_accuracy;
        cv_result.max_accuracy = (single_result > cv_result.max_accuracy) ? single_result : cv_result.max_accuracy;
        cv_result.accuracy_variance += (single_result - cv_result.avg_accuracy) * (single_result - cv_result.avg_accuracy);
    }
    return cv_result;
}

int main(int argc, char const *argv[])
{
    vector<vector<float>> mails_dataset = read_dataset("spambase.data");
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(mails_dataset.begin(), mails_dataset.end(), g);
    steady_clock::time_point begin = steady_clock::now();
    cross_validation_result result = __10_folds_cross_validation(mails_dataset);
    steady_clock::time_point end = steady_clock::now();
    double elapsedTime = static_cast<double>(duration_cast<microseconds>(end - begin).count()) / 1000000;
    cout << "Elapsed time = " << elapsedTime << " seconds.";
    result.print_scores();
    return 0;
}
