#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <random>
#include <iterator>

using namespace std;
using namespace std::chrono;

constexpr float ABSOLUTE_MIN = numeric_limits<float>::min();
constexpr float ABSOLUTE_MAX = numeric_limits<float>::max();

/**
 * @brief The following class defines the variables needed to include the most
 * important statistics of accuracy for evaluation.
 * This class was made extern in order to be accessible by Python.
 *
 */
extern "C"
{
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
            cout << "==========================================================" << endl
                 << "Naive bayes classification: " << endl
                 << "Minimum Accuracy: " << this->min_accuracy << endl
                 << "Average Accuracy: " << this->avg_accuracy << endl
                 << "Maximum Accuracy: " << this->max_accuracy << endl
                 << "Variance of Accuracy: " << this->accuracy_variance << endl
                 << "Standard Deviation of Accuracy: " << sqrt(this->accuracy_variance) << endl
                 << "==========================================================" << endl;
        }
    };
}

/**
 * @brief This class was defined in order to include the two functions needed
 * by cross validation to fit and test the training set for every fold.
 *
 */
class naive_bayes_classifier
{
public:
    naive_bayes_classifier() = default;
    ~naive_bayes_classifier() = default;

    vector<float> spam_mean,
        ham_mean,
        spam_variance,
        ham_variance;
    float spam_probability,
        ham_probability;

    /**
     * @brief The following function performs the test-scoring on the given test set.
     *
     * @param test_set
     * @return float is the score evaluated
     */
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

    /**
     * @brief This function performs the fitting of the model on the given dataset by
     * computing the means, the variances and the probability of the two classes.
     *
     * @param dataset
     */
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

/**
 * @brief This function prints the whole dataset.
 *
 * @param dataset
 */
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

/**
 * @brief This function reads the dataset from the given file path.
 *
 * @param filePath
 * @return vector<vector<float>>
 */
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
 * @brief The following function performs a ten fold cross-validation on the given dataset,
 * slicing in 10 folds the dataset, and then iterating over each fold, taking it as test set
 * and the rest as training set.
 *
 * @param mails_dataset is the dataset.
 * @return cross_validation_result is the object containing the scores of the cross validation.
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
    vector<vector<float>>::const_iterator dataset_begin = mails_dataset.begin();
    vector<float> all_scores(10);
    naive_bayes_classifier classifier;
    // CROSS VALIDATION.
    for (uint8_t validation_iteration = 0; validation_iteration < 10; validation_iteration++)
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
        classifier.fit(train_folds);
        all_scores[validation_iteration] = classifier.score(test_fold);
        total_score += all_scores[validation_iteration];
    }
    // pool.shutdown();
    cross_validation_result cv_result;
    cv_result.avg_accuracy = total_score / 10.;
    for (float &single_result : all_scores)
    {
        cv_result.min_accuracy = (single_result < cv_result.min_accuracy)
                                     ? single_result
                                     : cv_result.min_accuracy;
        cv_result.max_accuracy = (single_result > cv_result.max_accuracy)
                                     ? single_result
                                     : cv_result.max_accuracy;
        cv_result.accuracy_variance += (single_result - cv_result.avg_accuracy) *
                                       (single_result - cv_result.avg_accuracy);
    }
    return cv_result;
}

/**
 * @brief The following extern function reads and shuffles the dataset, and
 * eventually applies cross validation to it, returning the resulting scores
 * as cross_validation_result object.
 * This function was made extern in order to be accessible by Python.
 *
 */
extern "C"
{
    cross_validation_result py_main()
    {
        vector<vector<float>> mails_dataset = read_dataset("spambase.data");
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(mails_dataset.begin(), mails_dataset.end(), g);
        return __10_folds_cross_validation(mails_dataset);
    }
}

int main()
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
