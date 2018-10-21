import numpy as np
import matplotlib.pyplot as plt 

def calculate_predicted_labels(x, w, val=0):
    y_pred = x.dot(w)
    y_pred[np.where(y_pred <= val)] = -1
    y_pred[np.where(y_pred > val)] = 1
    
    return y_pred

def print_accuracy(predict_labels, x, y, train=True):
    total_correct_labels = np.sum(predict_labels == y)
    print('Total correct labels in training: {}'.format(total_correct_labels))
    if train:
        print('Training accuracy: {}'.format((total_correct_labels / x.shape[0]) * 100))
    else:
        print('Testing accuracy: {}'.format((total_correct_labels / x.shape[0]) * 100))

def predic(n):
    """
        use tanh to get the label(-1/1) of predic result n
        
    """
    n = np.tanh(n)
    n[n>=0] = 1
    n[n<0] = -1
    return n

def build_poly(x, degree):
    """
        polynomial basis functions for input data x, for j=0 up to j=degree.
        
        Input:
            x: feature data
            degree: the largest expoent of x need to be concated
        
        output:
            poly: the matrix with new features (shape should be [x.shape[0], 1+ degree * x.shape[1])
        
    """
    # ***************************************************
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    # ***************************************************
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def sigmoid(t):
    """
        Compute the sigmoid funtion to implement logistic regression
    """
    return 1 / (1 + np.exp(-t))

def compute_loss(y, tx, w, loss_function='mse'):
    """
        Calculate the loss.
        
        Input:
            y: weight
            tx: 1+height
            loss_function: mse, rmse, mae
            
        Output:
            loss
    
    """
    if loss_function == 'mse': # Mean square error
        e = y - tx.dot(w)
        return 1/2*np.mean(e ** 2)
   
    elif loss_function == 'rmse': # Square root of MSE
        e = y - tx.dot(w)
        return np.sqrt(1/2 * np.mean(e ** 2))
   
    else: # Mean Absolute Error
        e = y - tx.dot(w)
        return 1/2 * np.mean(np.abs(e))
    

def compute_gradient(y, tx, w):
    """
        compute gradient of loss
        
        :param: y: labels for the dataset
        :param: tx: input feature data
        
        output: gradient
    """
    N = len(y)
    error = y - np.dot(tx, w)
    gradient = -1/N*(tx.T.dot(error))
    return gradient

def least_squares_GD(y, tx, initial_w, max_iters, gamma, loss_function='mse'):
    """
        Gradient descent algorithm.
        
        :param: y: labels for the dataset
        :param: tx: input feature data
        :param: max_iters: number of iteration
        :param: gamma: step size for renew parameter
        :param: loss_function: Choose method to compute loss
        
        output: w, loss
    """
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        #  compute gradient and loss
        # ***************************************************
        loss = compute_loss(y, tx, w, loss_function)
        gradient = compute_gradient(y, tx, w)
        # ***************************************************
        #  update w by gradient
        # ***************************************************
        w = w - gamma * gradient
        # store w and loss
        ws.append(w)
        losses.append(loss)
        
        #print(ws, losses)
        
    return ws[-1], losses[-1]

def least_squares_SGD(y, tx, initial_w, max_iters, gamma, loss_function='mse'):
    """
        Implement Stochastic gradient descent algorithm. batch_size=1
        
        :param: y: labels for the dataset
        :param: tx: input feature data
        :param: max_iters: number of iteration
        :param: gamma: step size for renew parameter
        :param: loss_function: Choose method to compute loss
        
        output: w, loss
    """
    batch_size = 1
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient and loss
            grad = compute_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            loss = compute_loss(y, tx, w, loss_function=loss_function)
            # store w and loss
            ws.append(w)
            losses.append(loss)
            
            print('Loss: {}, iteration: {}'.format(loss, n_iter))

    return ws[-1], losses[-1]

def least_squares(y, tx, loss_function='mse'):
    """
        calculate the least squares solution.
        
        :param: y: labels for the dataset
        :param: tx: input feature data
        
        output: w, loss
    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w, loss_function)
    return w, loss

def ridge_regression(y, tx, lambda_):
    """
        implement ridge regression with L2 param lambda_.
        
        :param: y: labels for the dataset
        :param: tx: input feature data
        :param: lambda_: L2 nomalization param
        
        output: w, loss
        
    """
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w= np.linalg.solve(a, b)
    return w, compute_loss(y, tx, w)

def split_data(x, y, ratio, seed=1):
    """
        split the dataset based on the split ratio. If ratio is 0.8
        you will have 80% of your data set dedicated to training
        and the rest dedicated to testing
        
        :param: x: input data which is standardized and cleaned.
        :param: y: labels for the dataset
        :param: ratio: the ratio of train set in x and y
        :param: seed: random seed to split set defaut as 1
        
        output:x_train, x_test, y_train, y_test
    """
    # set seed
    np.random.seed(seed)
    num = len(y)
    order = np.random.permutation(num)
    order_1 = order[:int(np.floor(ratio*num))]
    order_2 = order[int(np.floor(ratio*num)):num]
    x_train = x[order_1]
    y_train = y[order_1]
    #ids_train = ids[order_1]
    x_test = x[order_2]
    y_test = y[order_2]
    #ids_test = ids[order_2]
    return x_train, x_test, y_train, y_test

def cross_validation(x, y, k=10, seed=1):
    """
    Split the data into k sets where k-1 sets is used for training
    and the kth set is used for testing.

    :param: x: input data
    :param: y: testing labels
    :param: k: parameter as to how many splits should be made
    """
    np.random.seed(seed)

    # Get the shuffled order of indices randomly
    shuffled_order = np.random.permutation(len(y))
    shuffled_x = x[shuffled_order]
    shuffled_y = y[shuffled_order]

    # Divide the input data and labels into k sets
    n = int(len(y) / k)
    x_k_sets = [x[i:i+n] for i in range(0, len(x), n)] 
    y_k_sets = [y[i:i+n] for i in range(0, len(y), n)]

    # For each value of k get the kth set and 
    for j in range(k):
        x_test, y_test = x_k_sets[j], y_k_sets[j]
        x_train = x_k_sets[np.arange(len(x_k_sets)) != j]
        y_train = y_k_sets[np.arange(len(y_k_sets)) != j]
        yield x_test, y_test, x_train, y_train

    # TODO: Right now just splits the dataset need to follow up
    # with a conceptual question

def standardize(x):
    """
        Standardize the original data set.
        Multiple columns represent features so need to take mean of each standardized feature
        
        :param: x: input data which needs to be standardized
        
        output: a tuple of x, mean_x, std_x
        
    """
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x


def build_model_data(x, y):
    """
        Form (tx, y) to get regression data in matrix form. Concat a 1 column to x.
        
        :param: x: input data which is standardized and cleaned.
        :param: y: labels for the dataset
        
        output: tx, y
    """
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return tx, y

def logistic_regression(y, tx, max_iters, gamma):
    """
    Implementing logistic regression algorithm.


    :param: tx: input data which is standardized and cleaned.
    :param: y: labels for the dataset
    :param: max_iters: times which the algorithm needs to be run.
    :param: gamma: the learning rate value.
        
    output: weight, loss
    """

    initial_w = np.zeros(tx.shape[1])
    divide_by_constant = 1 / y.shape[0]
    
    for n_iter in range(max_iters):
        h = sigmoid(np.dot(initial_w, tx.T))
        gradient = divide_by_constant * np.dot(tx.T, (h - y))
        initial_w -= gamma * gradient
        
        loss = calculate_loss_logistic(h, y)

        print(
            'Loss calculated at: {} , training step: {}'.format(
                 loss, n_iter
            )
        )
    return initial_w, loss
        
def reg_logistic_regression(y, tx, lambda_, max_iters, gamma):
    """
    Implementing logistic regression algorithm.


    :param: tx: input data which is standardized and cleaned.
    :param: y: labels for the dataset
    :param: max_iters: times which the algorithm needs to be run.
    :param: gamma: the learning rate value.
        
    output: weight, loss
    """

    initial_w = np.zeros(tx.shape[1])
    divide_by_constant = 1 / y.shape[0]
    
    for n_iter in range(max_iters):
        h = sigmoid(np.dot(initial_w, tx.T))
        constant = lambda_ / y.shape[0]
        gradient = (divide_by_constant * np.dot(tx.T, (h - y))) + (constant * initial_w)
        initial_w -= gamma * gradient
        
        loss = calculate_loss_logistic(
            h, y, initial_w, lambda_=lambda_, regularize=True
        )

        print(
            'Loss (regularization) calculated at: {} , training step: {}'.format(
                 loss, n_iter
            )
        )
    return initial_w, loss

def calculate_loss_logistic(h, y, w, lambda_=0, regularize=False):
    """
    Given the actual label y and calculated hypothesis h returns the loss
    accumulated over all data points.
    """
    loss = (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    if not regularize:
        return loss
    else: # lambda_/ 2 * number of features by the sum of dot product
        constant = (lambda_ / ((2 * y.shape[0])))
        return loss + (constant * np.dot(w, w))
