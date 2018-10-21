from proj1_helpers import *
from implementations import *

## Load data ##
y, x, ids = load_csv_data("train.csv")
x[x == -999] = 0
x, _, _ = standardize(x)

# build polynomial by degree
degree = 1
# split train dataset to 2 parts for test and train
ratio_split = 0.8

## Feature Engineering ##
x_ = build_poly(x, degree)


def run_logistic_regression(x_train, y_train, x_test, y_test, initial_w):
    ## Logistic Regression ##
    max_iters_logistic = 100
    lr = 0.001
    w, loss = logistic_regression(y_train, x_train, max_iters_logistic, lr, initial_w)

    # ## Calculate the number of right labels for training and testing ##
    training_predict_labels = calculate_predicted_labels(x_train, w, val=0.5)
    testing_predict_labels = calculate_predicted_labels(x_test, w, val=0.5)

    ## Print the logs of the final results ##
    print_accuracy(training_predict_labels, x_train, y_train)
    print_accuracy(testing_predict_labels, x_test, y_test, train=False)

    return w, loss

## Split data or do cross validation ##
split_data = False
if split_data:
    x_train, x_test, y_train, y_test = split_data(x_, y, ratio_split)
    print('The size of training data: {}\nThe size of training data: {}'.format( x_train.shape, x_test.shape))
    run_logistic_regression(x_train, y_train, x_test, y_test)
else:
    iter_n = 1
    cross_val_losses = []
    initial_w = np.zeros(x_.shape[1])
    for x_train, y_train, x_test, y_test in cross_validation(x_, y):
        print(
            'Iteration: {} The size of training data: {}'
            '\nThe size of testing data: {}'.format(
                iter_n, x_train.shape, x_test.shape
            )
        )

        weights, loss = run_logistic_regression(x_train, y_train, x_test, y_test, initial_w)

        # Update variables for next iteration
        initial_w = weights
        cross_val_losses.append(loss)
        iter_n += 1

    print('Losses over 10 cross validation cycles: {}'.format(cross_val_losses))
