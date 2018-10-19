import numpy as np
import matplotlib.pyplot as plt 

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
	#Compute the sigmoid funtion to implement logistic regression
	t = np.divide(1, 1+np.exp((-1)*t))
	return t

def compute_loss(y, tx, w):
    """Calculate the loss.
    y: weight
    tx: 1+height
    """
    # ***************************************************
    # compute loss by MSE
    # ***************************************************

    e = y - tx.dot(w)

    return 1/2*np.mean(e**2)

def compute_gradient(y, tx, w):
    # ***************************************************
    #  compute gradient and loss
    # ***************************************************
    N = len(y)
    error = y - np.dot(tx, w)
    gradient = -1/N*(tx.T.dot(error))
    return gradient

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        #  compute gradient and loss
        # ***************************************************
        loss = compute_loss(y, tx, w)
        gradient = compute_gradient(y, tx, w)
        # ***************************************************
        #  update w by gradient
        # ***************************************************
        w = w - gamma*gradient
        # store w and loss
        ws.append(w)
        losses.append(loss)
    return ws[-1], losses[-1]

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm.fdsaf"""
    # ***************************************************
    # Implement stochastic gradient descent.
    # **************************************************
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
            loss = compute_loss(y, tx, w)
            # store w and loss
            ws.append(w)
            losses.append(loss)

    return ws[-1], losses[-1]

def least_squares(y, tx):
    """calculate the least squares solution."""
    # ***************************************************
    # returns weights
    # ***************************************************
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    return w, loss

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w= np.linalg.solve(a, b)
    return w, compute_loss(y, tx, w)

def split_data(x, y, ids, ratio, seed=1):
    """
        split the dataset based on the split ratio. If ratio is 0.8
        you will have 80% of your data set dedicated to training
        and the rest dedicated to testing
        """
    # set seed
    np.random.seed(seed)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # split the data based on the given ratio: TODO
    # ***************************************************
    num = len(y)
    order = np.random.permutation(num)
    order_1 = order[:int(np.floor(ratio*num))]
    order_2 = order[int(np.floor(ratio*num)):num]
    x_train = x[order_1]
    y_train = y[order_1]
    ids_train = ids[order_1]
    x_test = x[order_2]
    y_test = y[order_2]
    ids_test = ids[order_2]
    return x_train, x_test, y_train, y_test, ids_train, ids_test

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    return 0
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    return 0
