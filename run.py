from proj1_helpers import *
from implementations import *
from categroy import *

# Initializing the parameters for each category based on JET number
c_parameters = dict()
##------------------poly_deg----comb_size----K_fold----step_size----random_seed----iteration----
c_parameters[0] = [   2,          1,         8,          0.001,           1,             8000, False]
c_parameters[1] = [   2,          2,         8,          0.001,           1,             8000, False]
c_parameters[2] = [   2,          2,         10,         0.0001,          1,            45000, False]
c_parameters[3] = [   2,          2,         10,         0.001,           1,            10000, False]

## Load data ##
x_cate, y_cate, ids_cate, total_num = load_categrized_data("train.csv")
x_cate_rslt, y_cate_rslt, ids_cate_rslt, total_num_rslt = load_categrized_data("test.csv")

#####################
#get prediction model
#####################
category_weights = dict()
predic_result = dict()

lambda_ = 5 # Regularization Parameter
for cate_num in range(4):

    print('Training the model for category: {}'.format(cate_num))
    
    poly_degree, comb_size, k_fold, gamma, seed, max_iters, do_decay = c_parameters[cate_num]
    print('This is the gamma used: {}'.format(gamma))

    cross_val_weights = []
    loss_sum = 0
    right_pred_fold = 0
    best_accur = 0

    x_t, y_t = x_cate[cate_num], y_cate[cate_num]
    x_te = x_cate_rslt[cate_num]
    k_indices = build_k_indices(y_t, k_fold, seed)
    
    print('################################################################')

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

    # Initialize the weights
    initial_w = np.random.randn(x_t.shape[1])
    
    # Pass in cross validated
    # w, loss = logistic_regression(y_t, x_t, max_iters, gamma, initial_w, do_decay=do_decay)
    w, loss = reg_logistic_regression(y_t, x_t, lambda_, max_iters, gamma, initial_w)
    loss_sum += loss

    # Get training accuracy
    training_predict_labels = calculate_predicted_labels(x_t, w, val=0.5, do_sigmoid=True)
    print_accuracy(training_predict_labels, y_t, train=True)

    # Get testing accuracy
    testing_predict_labels = calculate_predicted_labels(x_te, w, val=0.5, do_sigmoid=True)
    predic_result[cate_num] = testing_predict_labels

    # find the best weight
    # if (accur/ len(y_te) > best_accur/ len(y_te)):
    #     best_accur = accur
    category_weights[cate_num] = w

    print("The Average loss of train set: {}".format(loss_sum))
    print("Shape of weight: ", category_weights[cate_num].shape)
    print('Training for CATEGORY DONE: ########################################################')


# print weights to file
f = open("category_weights.txt", 'w+')    
for i in range(4):
    print(category_weights[i], file=f)

#output result and sort them by IDs
result_y = np.array([])
result_ids = np.array([])
for cate_num in range(4):
    result_y = np.r_[result_y, predic_result[cate_num]]
    result_ids = np.r_[result_ids, ids_cate_rslt[cate_num]]
create_csv_submission(result_ids, result_y, 'test_predicted.csv')
print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
