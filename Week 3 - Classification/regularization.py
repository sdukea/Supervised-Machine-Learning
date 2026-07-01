import numpy as np
import matplotlib.pyplot as plt

# If one way to reduce overfitting is 
# to add more training examples, then why is another way, 
# to reduce overfitting, regularization or reduce the number/impact of features we have? 
# Isn't that counter-intuitive?

# the problem in overfitting:
# - flexibility: how easily the model can bend itself to fit data
# - a very flex. model can create extremely complicated decision boundaries or curves
# More flexibility means:
# better ability to capture complex real patterns
# but also better ability to memorize noise

# your tr. data is the only evidence your model has about the real world
# but data is: limited, noisy and incomplete
# with only a few examples, there is little-to-no evidence of the underlying pattern between our
# feature values/vector vs target output

# and because of this, a highly flexible model may start believing that the
# random fluctuations/noise in the training data are the actual patterns
# this happens because we do not yet have enough data/evidence
# to reliably distinguish:
# - true signal
# - from random coincidence/noise

# with only a small dataset, noise can accidentally look like
# a meaningful pattern

# but with much larger amounts of data:
# - true patterns tend to appear consistently
# - random noise becomes less consistent/statistically weaker

# therefore, even when noise exists,
# the underlying relationship becomes easier to detect

# so more data helps the model separate:
# signal vs noise

# but NOTE: you're still trying to fit a complex, flexible model to this highly evident
# data - you still have chances of overfitting
# and that's why you have to reduce model complexity as well

# one way to think: overfitting = model complexity / amount of data

# so we can solve overfitting by increasing denominator as it reduces the amount of overfitting
# (smaller denom., smaller overfitting)
# |
# and one more way is also to decrease numerator – decrease model complexity

# and this is exactly what explains regularization

# you have added more training examples to the data - that's cool
# you now have an improved visibility of the signal
# but this larger dataset still has noise/inconsistency/fluctuations

# more data makes the underlying pattern/generalizable pattern easier to detect by
# the model

# but a highly flexible, complex model may STILL learn the noise/fluctations in this pattern-evident
# data

# so you have to ensure your model isn't complex

# and that's why you regularize – you encourage the model to prefer smoother/simpler relationships
# without trying to learn noise and only stick to signal/generalizable pattern in the data

# more data: increases signal/more correct evidence
# regularization: controls model complexity/flexibility

# and you reduce model complexity by reducing the impact of parameters w1 to wn
# essentially, we penalize/regularize the model if associated parameters are huge - we minimize the 
# size of these parameters close to 0

# larger weights: feature is important
# lower weights: feature is not important

# so, weights -> gradients (when illustrated in graphs)
# larger weights for a specific feature, more the slope it has
# and larger the slope, sharper the bends, more aggressive the feature influence

# so extreme sensitvity, steep slopes and aggressive reactions are mathematically caused by 
# very large weights - feature influence

# and that's why we have to minimize them to stop overfitting


# –––

# adding regularization

# cost functions with reg.

# c.f. for regularized linear reg.

def compute_cost_linear_reg(X, y, w, b, lambda_=1.0):

    m = X.shape[0]

    n = len(w)

    cost = 0

    for i in range(m):
        f_wb_i_vec = np.dot(X[i], w) + b

        cost = cost + (f_wb_i_vec - y[i])**2

    cost = cost / (2*m)

    # if prediction error is high, cost is high and hence we minimize cost

    reg_cost = 0

    for j in range(n):
        reg_cost += (w[j]**2)
    
    reg_cost = (lambda_/(2*m)) * reg_cost

    # if the weights for any feature is large, cost adds up and becomes high and hence we should
    # minimze it (penalization)
    
    # so the optimization algorithm (G.D./Adam) shows us two goals:
    # 1. predict correctly
    # 2. keep weights small

    # if w = [1, 2]
    # penalty = 1^2 + 2^2 = 5
    # smaller penalty, cost adds up slightly - showing chosen w is somewhat optimal

    # but if w = [100, 200]
    # penalty = 100^2 + 200^2 = 50000
    # larger penalty, cost adds up by a lot - showing the chosen w is not optimal

    # squaring the weights is exactly like squaring the pred. error of f_wb_x_i - y[i]
    # Because we want the cost function to treat very large weights as dramatically worse 
    # than moderately large weights
    # and same for prediction error; the cost function should treat large prediction errors as
    # dramatically worse than moderately large pred. errors

    # you might think: why not just use the weight values by itself to penalize the
    # if w = [-100, 100]
    # then penalty = -100 + 100 = 0
    # this shouldn't be the case
    # that's why we square them to get the actual picture of penalty and added cost
    # so that our optimization algorithm can evidently minimize cost
    # (-100)^2 + 100^2 = gives you the actual picture of the penalty and cost that is added

    total_cost = cost + reg_cost

    return total_cost

# let's see this in action

np.random.seed(1)

X_tmp = np.random.rand(5, 6)
# generate a 5 x 6 matrix filled with random numbers from 0 to 1
# every re-run -> same numbers

y_tmp = np.array([0, 1, 0, 1, 0])

w_tmp = np.random.rand(X_tmp.shape[1]).reshape(-1) - 0.5
# just a random way to get a set of parameters w1 to wn (as n = X_tmp.shape[1])

b_tmp = 0.6

lambda_tmp = 0.7

cost_tmp = compute_cost_linear_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print('Regularized cost:', cost_tmp)

# c.f. for regularized logistic regression

def sigmoid(z):

    g_z = 1 / (1 + np.exp(-z))

    return g_z

def compute_cost_log_reg(X, y, w, b, lambda_=1.0):

    m, n = X.shape

    cost = 0

    for i in range(m):

        z_i = np.dot(X[i], w) + b

        f_wb_i = sigmoid(z_i)

        cost += -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1 - f_wb_i)

    cost = cost / (2*m)

    reg_cost = 0

    for j in range(n):

        reg_cost += (w[j]**2)
    
    reg_cost = (lambda_/(2*m)) * reg_cost

    total_cost = cost + reg_cost

    return total_cost

# see it in action

log_cost_tmp = compute_cost_log_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print(f"Regularized cost (logistic): {log_cost_tmp}")

# gradient descent for regularized linear regression

# g.d. for reg. linear regression is around this idea

# J = prediction error + regularization penalty

# now, when you differentiate it, you get

#  (first term/usual gradient)       (second term)
# = d(prediction error)/dwj + d(regularization penalty)/dwj

# so you have the new addition ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# the first term says: change wj to fit the data better

# the second term says: reduce wj if its large

# Both terms modify the same parameter(s) w (w1 to wn) for different
# objectives

# without regularization: the only thing wj is aimed at is to fit data better
# this can easily result in overfitting because the only goal for the model is to fit
# data better/reduce prediction error for every training example

# say w1 = 10, a = 0.1, lambda = 2, m = 100 (an example)
# also, say usual gradient = -5

# normal update => w1 = w1 - alpha * usual gradient
#                     = 10 - 0.1(-5)
#                     = 10.5

# the weight value has increased so as to fit the training data well - reduce prediction error

# now, this is fine, but again - this is prone to overfitting as you are only
# trying to fit the training data - reduce pred. error for all points/learn noise
# this introduces overfitting and the only way you can reduce this is to

# regularize - reduce impact of features
# i.e. to prioritize reducing the size of parameters

# say m = 100

# second term = lambda * (w1/m) = 2 * (10/100) = 0.2

# total gradient = usual gradient + second term
#                = -5 + 0.2 = -4.8

# regularized update => w1 = w1 - total gradient
#                          = 10 - 0.1(-4.8)
#                          = 10.48

# what you've done now:
# 1. you have made the model fit the training data to a good extent so that it does not overfit/
# generalizes better/does not learn noise
# 2. prioritized reducing the size of parameters so as to reduce overfitting

# You will have to priortize both
# and that is the tradeoff regularization intentionally creates

# I am willing to accept a slightly worse fit to the training data if it gives me a 
# simpler model that generalizes better.

# Overfitting happens precisely because the model is sometimes too focused on 
# fitting the training data - that's why we take its attention away from just trying to
# reduce pred. error on each tr. eg. and fit the training data well to reduce the size of
# parameters and produce a simpler, smoother model

# that's the idea

def compute_gradient_linear_reg(X, y, w, b, lambda_):

    m, n = X.shape

    dj_dw = np.zeros((n,))
    dj_db = 0

    for i in range(m):

        pred = np.dot(X[i], w) + b

        error = (pred - y[i])**2

        for j in range(n):

            dj_dw[j] = dj_dw[j] + error * X[i, j]

        dj_db += error
    
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    for j in range(n):

        dj_dw[j] = dj_dw[j] + (lambda_/m) * w[j]

    return dj_db, dj_dw


dj_db_tmp, dj_dw_tmp = compute_gradient_linear_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print(dj_db_tmp, dj_dw_tmp)

# gradient descent for regularized logistic regression


def compute_gradient_log_reg(X, y, w, b, lambda_):

    m, n = X.shape

    dj_dw = np.zeros((n,))

    dj_db = 0.0

    for i in range(m):
        z_i = np.dot(X[i], w) + b

        f_wb_x_i = sigmoid(z_i)

        error = (f_wb_x_i - y[i])**2

        for j in range(n):
            dj_dw[j] = dj_dw[j] + error * X[i, j]

        dj_db += error

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    # done for the first term; prediction error

    for j in range(n):

        dj_dw[j] = dj_dw[j] + (lambda_/m) * w[j]

    return dj_db, dj_dw