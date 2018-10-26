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


c_parameters = dict()
##------------------poly_deg----comb_size----K_fold----step_size----random_seed----iteration----
c_parameters[0] = [   2,          1,         8,          0.001,           1,             8000, False] # 84.4%
c_parameters[1] = [   2,          2,         8,          0.001,           1,             8000, False] # 80 - 81%
c_parameters[2] = [   2,          2,         10,         0.0001,          1,            45000, False] # Reduced the number of iterations from 80k to 45k
c_parameters[3] = [   2,          2,         10,         0.001,           1,            10000, False]

## Load data ##
x_cate, y_cate, ids_cate, total_num = load_categrized_data("train.csv")

x_cate_rslt, y_cate_rslt, ids_cate_rslt, total_num_rslt = load_categrized_data("test.csv")

# Remove columns which have high correlation - refer from notebook ##
# Delete these features since they are correlated with DER_deltaeta_jet_jet and have cyclic correlation
# DER_lep_eta_centrality, DER_prodeta_jet_jet, PRI_jet_subleading_eta, PRI_jet_subleading_phi, PRI_jet_subleading_pt
# for i in [6, 11, 24, 24, 24]:
#     x = np.delete(x, i, 1)


# Replace -999 values with the mean
# for column in range(x.shape[1]):
#     x[:, column] = replace_nan(x[:, column], -999)

# x, _, _ = standardize(x)

# for cate_num in range(4):

#####################
#get prediction model
#####################
# right_pred_num = 0
# total_pred_num = 0
category_weights = dict()
predic_result = dict()

lambda_ = 5
for cate_num in range(4):

    print('Training the model for category: {}'.format(cate_num))
    
    poly_degree, comb_size, k_fold, gamma, seed, max_iters, do_decay = c_parameters[cate_num]
    print('This is the gamma used: {}'.format(gamma))
## Split data or do cross validation ##
# do_split_data = False

# if do_split_data:
#     # split train dataset to 2 parts for test and train
#     ratio_split = 0.8
#     x_train, x_test, y_train, y_test = split_data(x_, y, ratio_split)

#     print('The size of training data: {}\nThe size of training data: {}'.format( x_train.shape, x_test.shape))
#     run_logistic_regression(x_train, y_train, x_test, y_test)
# else:

    # cross_val_losses = []
    cross_val_weights = []
    loss_tr_sum = 0
    right_pred_fold = 0
    best_accur = 0
    # split_data = False
    # # Split the dataset for cross validation and testing
    # if split_data:
    #     x_train, x_test, y_train, y_test = split_data(x_, y_cate[cate_num], 0.9)
    # else:
    x_t, y_t = x_cate[cate_num], y_cate[cate_num]
    x_te = x_cate_rslt[cate_num]
    k_indices = build_k_indices(y_t, k_fold, seed)
    
    # for k in range(k_fold):

        # Cross validated training set
    print('################################################################')
    # x_t, y_t, x_te, y_te = cross_validation(y_train, x_train, k_indices, k)
    
    # Do feature engineering

    x_t = build_combination(x_t, comb_size)
    print('Build features for the poly-combination task: {}'.format(x_t.shape))
    x_te = build_combination(x_te, comb_size)
    print('Build features for the poly-combination task: {}'.format(x_te.shape))

    x_t = build_sqrt(x_t)
    print('Build features for the sqrt task: {}'.format(x_t.shape))
    x_te = build_sqrt(x_te)
    print('Build features for the sqrt task: {}'.format(x_te.shape))

    x_t = build_log(x_t)
    print('Build features for the log task: {}'.format(x_t.shape))
    x_te = build_log(x_te)
    print('Build features for the log task: {}'.format(x_te.shape))



    # Standardize it
    x_t, x_te = standardize(x_t, x_te)

    
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
    predic_result[cate_num] = testing_predict_labels
    # accur = print_accuracy(testing_predict_labels, y_te, train=False)
    # right_pred_fold += accur
    #cross_val_weights.append(w)

    #find the best weight
    # if (accur/ len(y_te) > best_accur/ len(y_te)):
    #     best_accur = accur
    category_weights[cate_num] = w


    print("The Average loss of train set: {}".format(loss_tr_sum))#/k_fold))
    # print("The Average accuarcy of test set in category {} is {}.".format(
    #     cate_num, 100 * right_pred_fold / (len(y_te)), '%')
    # )
    # print("The BEST accuarcy of test set in category {} is {}.".format(
    #     cate_num, 100 * best_accur/ len(y_te), '%')
    # )
    print("Shape of weight: ", category_weights[cate_num].shape)
    print('Training for CATEGORY DONE: ########################################################')
    #mean_of_weights = np.mean(cross_val_weights, axis=0)
    # Add the weight for this category
    #category_weights.append(mean_of_weights)

    # testing_predict_labels = calculate_predicted_labels(x_test, mean_of_weights, val=0.5, do_sigmoid=True)
    # total_correct_testing_labels = print_accuracy(testing_predict_labels, y_test, train=False)
    # print('The total accuracy of testing data: {}'.format(100 * (total_correct_testing_labels / len(y_test))))

#print w to file
f = open("category_weights.txt", 'w+')    
for i in range(4):
    print(category_weights[i], file=f)

#output result
result_y = np.array([])
result_ids = np.array([])
for cate_num in range(4):
    result_y = np.r_[result_y, predic_result[cate_num]]
    result_ids = np.r_[result_ids, ids_cate_rslt[cate_num]]
create_csv_submission(result_ids, result_y, 'test_predicted.csv')
print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
sort_csv_byColumn('test_predicted.csv', 'test_predicted_ordered.csv', 'Id') ##pd.read_csv('test_predicted.csv').sort_values(['Id']).to_csv(test_predicted_ordered.csv)
