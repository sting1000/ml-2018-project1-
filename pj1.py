%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt #for plot

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
    MSE = np.sum(np.power((y - np.dot(tx, w)), 2))/(2*len(y))
    return MSE

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
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return ws[-1], losses[-1]

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # ***************************************************
    # Implement stochastic gradient descent.
    # ***************************************************
    losses = []
    batch_size = 1
    ws = [initial_w]
    w = initial_w
    for item in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            loss = compute_loss(minibatch_y, minibatch_tx, w)
            gradient = compute_gradient(minibatch_y, minibatch_tx, w)
            w = w - gamma*gradient
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

def logistic_regression(y, tx, initial_w, max_iters, gamma)
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)