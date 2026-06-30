import numpy as np
import matplotlib.pyplot as plt
import copy, math

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

b_init = 785.1811367994083
w_init = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])

def compute_cost(X, y, w, b):

    m = X.shape[0]

    cost = 0

    for i in range(m):
        f_wvecb_xvec_i = np.dot(X[i], w) + b

        error = (f_wvecb_xvec_i - y[i])**2

        cost += error

        print((1/2*m)*cost)

    cost = cost / (2*m)
    return cost

cost =  compute_cost(X_train, y_train, w_init, b_init)
print(f"Final cost: {cost}")

def compute_grad_terms(X, y, w, b):

    m, n = X.shape

    dj_dw = np.zeros((n,))

    dj_db = 0

    for i in range(m):

        f_wvecb_xvec_i = np.dot(X[i], w) + b

        error = (f_wvecb_xvec_i - y[i])**2

        for j in range(n):

            dj_dw[j] = error * X[i, j]

        dj_db += error

        dj_dw = dj_dw / m
        dj_db = dj_db / m
    
    return dj_dw, dj_db

dj_dw, dj_db = compute_grad_terms(X_train, y_train, w_init, b_init)

def gradient_descent(X, y, w_in, b_in, cost_func, grad_terms, alpha, num_iters):

    J_history = []

    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_dw, dj_db = grad_terms(X, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000:
            J_history.append(cost_func(X, y, w, b))

    return w, b, J_history

w_opt, b_opt, J = gradient_descent(X_train, y_train, w_init, b_init, compute_cost, 
                                   compute_grad_terms, alpha=5.0e-7, num_iters=1000)

