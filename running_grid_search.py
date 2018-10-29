from categroy import *
from proj1_helpers import *
from implementations import *


def run_reg_log_regression(cate_num, poly_degree, comb_size, k_fold, gamma, seed, max_iters, do_decay):

    print('Training the model for category: {}'.format(cate_num))
    print('This is the gamma used:'.format(gamma))
    cross_val_weights = []

    split_data = False
    # Split the dataset for cross validation and testing
    if split_data:
        x_train, x_test, y_train, y_test = split_data(x_, y_cate[cate_num], 0.9)
    else:
        x_train, y_train = x_cate[cate_num], y_cate[cate_num]


    k_indices = build_k_indices(y_train, k_fold, seed)
    loss_tr_sum = 0
    right_pred_fold = 0

    for k in range(k_fold):

        # Cross validated training set
        print('################################################################')
        x_t, y_t, x_te, y_te = cross_validation(y_train, x_train, k_indices, k)
        # Standardize it
        x_t, x_te = standardize(x_t, x_te)

        # Do feature engineering

        x_t = build_combination(x_t, comb_size)
        print('Build features for the poly-combination task: {}'.format(x_t.shape))
        x_te = build_combination(x_te, comb_size)
        print('Build features for the poly-combination task: {}'.format(x_te.shape))

        x_t = build_poly(x_t, poly_degree)
        print('Build features for the poly task (train shape): {}'.format(x_t.shape))
        x_te = build_poly(x_te, poly_degree)
        print('Build features for the poly task (test shape): {}'.format(x_te.shape))

        initial_w = np.random.randn(x_t.shape[1])

        # Pass in cross validated
        # w, loss = logistic_regression(y_t, x_t, max_iters, gamma, initial_w, do_decay=do_decay)
        w, loss = reg_logistic_regression(y_t, x_t, lambda_, max_iters, gamma, initial_w)
        loss_tr_sum += loss

        # Get training accuracy
        training_predict_labels = calculate_predicted_labels(x_t, w, val=0.5, do_sigmoid=True)
        print_accuracy(training_predict_labels, y_t, train=True)

        # Get testing accuracy
        testing_predict_labels = calculate_predicted_labels(x_te, w, val=0.5, do_sigmoid=True)
        right_pred_fold += print_accuracy(testing_predict_labels, y_te, train=False)

        cross_val_weights.append(w)

    print("The Average loss of train set: {}".format(loss_tr_sum/k_fold))
    print("The Average accuarcy of test set in category {} is {}.".format(
        cate_num, 100 * right_pred_fold / (len(y_te) * k_fold), '%')
    )
    print('Training for CATEGORY DONE: ########################################################')

    mean_of_weights = np.mean(cross_val_weights, axis=0)
    # Add the weight for this category
    category_weights.append(mean_of_weights)

    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')


## Load data ##
x_cate, y_cate, ids_cate, total_num = load_categrized_data("train.csv")

#####################
#get prediction model
#####################
right_pred_num = 0
total_pred_num = 0
category_weights = []

lambda_ = 5 # Regularization parameter


## Doing Grid Search ##

## In this we found out different hyperparameters for each of the categories
## We found out that below hyperparameters which do not overfit and provide us with high amount of accuracy
## Category 0 - 2 (polynominal degree), 1 (combination of size), 8 (k-fold), 0.001 (learning rate), 8000 (iterations)
## Category 1 - 2 (polynominal degree), 2 (combination of size), 8 (k-fold), 0.001 (learning rate), 8000 (iterations)
## Category 2 - 2 (polynominal degree), 2 (combination of size), 10 (k-fold), 0.0001 (learning rate), 45000 (iterations)
## Category 3 - 2 (polynominal degree), 2 (combination of size), 10 (k-fold), 0.001 (learning rate), 10000 (iterations)

## We are not doing grid search on higher degree of polynomials since it takes longer to train and the change in
## accuracy is minimal. Even with more features, the loss increases substantially due to the curse of dimensionality.

for i in range(4):
    for poly_degree in [1,2,3]:
        for comb_size in [1,2]:
            for gamma in [0.01, 0.001, 0.0001]:
                for max_iters in [4000,5000,8000,15000,20000,45000]:
                    run_reg_log_regression(i, poly_degree, comb_size, 10, gamma, 1, max_iters, False)
