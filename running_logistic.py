from proj1_helpers import *
from implementations import *
from categroy import *

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
    print_accuracy(training_predict_labels, y_train)
    print_accuracy(testing_predict_labels,  y_test, train=False)

    return w, loss



## Load data ##
x_cate, y_cate, ids_cate, total_num = load_categrized_data("train.csv")

# Remove columns which have high correlation - refer from notebook ##
# Delete these features since they are correlated with DER_deltaeta_jet_jet and have cyclic correlation
# DER_lep_eta_centrality, DER_prodeta_jet_jet, PRI_jet_subleading_eta, PRI_jet_subleading_phi, PRI_jet_subleading_pt
# for i in [6, 11, 24, 24, 24]:
#     x = np.delete(x, i, 1)


# Replace -999 values with the mean
# for column in range(x.shape[1]):
#     x[:, column] = replace_nan(x[:, column], -999)

# x, _, _ = standardize(x)

#initial w_cate, y_pred
w_cate = dict()
y_pred = dict()
for cate_num in range(4):
    w_cate[cate_num] = []
    y_pred[cate_num] = []


#####################
#get prediction model
#####################
right_pred_num = 0
total_pred_num = 0
for cate_num in range(4):
    x = x_cate[cate_num]
    #Parameters
    ### build polynomial by degree
    poly_degree = 2
    ### combination size
    comb_size = 1
    ### K fold validation
    k_fold = 5
    ### step size
    gamma = 0.1
    ###random seed
    seed = 1
    ###iteration
    max_iters = 500
## Split data or do cross validation ##
# do_split_data = False

# if do_split_data:
#     # split train dataset to 2 parts for test and train
#     ratio_split = 0.8
#     x_train, x_test, y_train, y_test = split_data(x_, y, ratio_split)

#     print('The size of training data: {}\nThe size of training data: {}'.format( x_train.shape, x_test.shape))
#     run_logistic_regression(x_train, y_train, x_test, y_test)
# else:
    
    ## Feature Engineering ##
    x_ = build_poly(x, poly_degree)
    x_ = build_combination(x_, comb_size)

    cross_val_losses = []
    cross_val_weights = []
    

    # Split the dataset for cross validation and testing
    x_train, x_test, y_train, y_test = split_data(x_, y_cate[cate_num], 0.9)

    k_indices = build_k_indices(y_train, k_fold, seed)
    loss_tr_sum = 0.
    #loss_te_sum = 0.
    right_pred_fold = 0
    for k in range(k_fold):
        w, loss_tr,_ = cross_validation2(y_train, x_train, k_indices, k, gamma, max_iters)
        loss_tr_sum += loss_tr
        #loss_te_sum += loss_te
        testing_predict_labels = calculate_predicted_labels(x_test, w, val=0.5, do_sigmoid=True)
        right_pred_fold += print_accuracy(testing_predict_labels, y_test, train=False)
        #accuarcy = 100 * print_accuracy(testing_predict_labels, y_test, train=False)/len(y_test)
        #print('The Accuarcy of fold {} in category {} is {}'.format(k, cate_num, accuarcy))
    print("The Average loss of train set: ", loss_tr_sum/k_fold)
    #print("The Average loss of test set: ", loss_te_sum/k_fold)
    print("The Average accuarcy of test set in category {} is {}: ".format(cate_num, 100*right_pred_fold/(len(y_test)*k_fold), '%'))
    print('########################################################')
    right_pred_num += int(right_pred_fold/ k_fold)
    total_pred_num += len(y_test)
print("Overall Accuarcy: ", 100*right_pred_num/total_pred_num,'%')

#     iter_n = 1
#     for x_train, y_train, x_test, y_test in cross_validation(x_train, y_train, k= k):
#         print(x_train[1])
#         initial_w = np.zeros(x_train.shape[1])
#         weights, loss = run_logistic_regression(x_train, y_train, x_test, y_test, initial_w, regularize=False)
#         # Update variables for next iteration
#         cross_val_weights.append(weights)
#         cross_val_losses.append(loss)
#         print(
#             'Iteration: {} The size of training data: {}'
#             '\nThe size of testing data: {}'.format(
#                 iter_n, x_train.shape, x_test.shape
#             ))
#         print('###############################################')
#         iter_n += 1
#     # print('Weights over 10 cross validation cycles: {}'.format(cross_val_weights))
#     print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
#     mean_of_weights = np.mean(np.array(cross_val_weights), axis=0)
#     print('Mean of weights: {} and shape: {}'.format(mean_of_weights, mean_of_weights.shape))
#     print('Losses over 10 cross validation cycles: {}'.format(cross_val_losses))

#     testing_predict_labels = calculate_predicted_labels(x_test, mean_of_weights, val=0.5, do_sigmoid=True)
#     right_pred_num += print_accuracy(testing_predict_labels, y_test, train=False)
#     total_pred_num += len(y_test)
#     w_cate[cate_num] = mean_of_weights
#     y_pred[cate_num] = testing_predict_labels
#     print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
# print(100*right_pred_num/total_pred_num,'%')