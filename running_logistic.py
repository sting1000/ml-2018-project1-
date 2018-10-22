from proj1_helpers import *
from implementations import *

## Load data ##
y, x, ids = load_csv_data("train.csv")

## Remove columns which have high correlation - refer from notebook ##
# Delete these features since they are correlated with DER_deltaeta_jet_jet and have cyclic correlation
# DER_lep_eta_centrality, DER_prodeta_jet_jet, PRI_jet_subleading_eta, PRI_jet_subleading_phi, PRI_jet_subleading_pt
for i in [6, 11, 24, 24, 24]:
    x = np.delete(x, i, 1)


# Replace -999 values with the mean
for column in range(x.shape[1]):
    x[:, column] = replace_nan(x[:, column], -999)

x, _, _ = standardize(x)

# build polynomial by degree
degree = 1

## Feature Engineering ##
x_ = build_poly(x, degree)
# x_ = build_combination(x_, 2)


def run_logistic_regression(x_train, y_train, x_test, y_test, initial_w, regularize=False):
    ## Logistic Regression ##
    max_iters_logistic = 500
    lr = 0.1
    lambda_ = 5

    if not regularize:
        w, loss = logistic_regression(y_train, x_train, max_iters_logistic, lr, initial_w)
    else:
        w, loss = reg_logistic_regression(y_train, x_train, lambda_, max_iters_logistic, lr, initial_w)    

    ## Calculate the number of right labels for training and testing ##
    training_predict_labels = calculate_predicted_labels(x_train, w, val=0.5, do_sigmoid=True)
    testing_predict_labels = calculate_predicted_labels(x_test, w, val=0.5, do_sigmoid=True)

    ## Print the logs of the final results ##
    print_accuracy(training_predict_labels, x_train, y_train)
    print_accuracy(testing_predict_labels, x_test, y_test, train=False)

    return w, loss

## Split data or do cross validation ##
do_split_data = False

if do_split_data:
    # split train dataset to 2 parts for test and train
    ratio_split = 0.8
    x_train, x_test, y_train, y_test = split_data(x_, y, ratio_split)

    print('The size of training data: {}\nThe size of training data: {}'.format( x_train.shape, x_test.shape))
    run_logistic_regression(x_train, y_train, x_test, y_test)
else:
    k = 10
    iter_n = 1
    cross_val_losses = []
    cross_val_weights = []
    initial_w = np.zeros(x_.shape[1])

    # Split the dataset for cross validation and testing
    x_train, x_test, y_train, y_test = split_data(x_, y, 0.9)

    for x_train, y_train, x_test, y_test in cross_validation(x_train, y_train, k=k):
        print(
            'Iteration: {} The size of training data: {}'
            '\nThe size of testing data: {}'.format(
                iter_n, x_train.shape, x_test.shape
            )
        )

        weights, loss = run_logistic_regression(x_train, y_train, x_test, y_test, initial_w, regularize=False)

        # Update variables for next iteration
        cross_val_weights.append(weights)
        initial_w = np.zeros(x_.shape[1])
        cross_val_losses.append(loss)
        iter_n += 1

    # print('Weights over 10 cross validation cycles: {}'.format(cross_val_weights))

    mean_of_weights = np.mean(np.array(cross_val_weights), axis=0)
    print('Mean of weights: {} and shape: {}'.format(mean_of_weights, mean_of_weights.shape))

    print('Losses over 10 cross validation cycles: {}'.format(cross_val_losses))

    testing_predict_labels = calculate_predicted_labels(x_test, mean_of_weights, val=0.5, do_sigmoid=True)
    print_accuracy(testing_predict_labels, x_test, y_test, train=False)
