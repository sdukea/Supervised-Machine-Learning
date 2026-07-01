# exploring situations where overfitting may occur
# and some solutions

import matplotlib.pyplot as plt

from ipywidgets import Output

# as implementation has helper functions, cannot see overfitting

# let's DIY

from sklearn.datasets import make_moons

# we're demonstrating overfitting on a dataset that has a complex relationship
# so data is like:
# two crescent moons side by side where one is facing upward and another facing downward

# so if we'd want to fit a model to this, we'd have to build a careful model (polynomial, of course)
# where there are high chances of overfitting

# that's why there's this example dataset
# 1. to be able to fit a complex polyn. model to the dataset
# 2. have high risk of overfitting so that we can see it and solve it



X, y = make_moons(
    n_samples=300,
    noise=0.25,
    random_state=42
)

# data:
# is two crescent moon clusters of data
# and each cluster of data/moon points correspond to one binary outcome 0 or 1
# one might be inverted or both might be facing the same direction
# data does NOT show a linear relationship - a line cannot separate them so
# that's why this is usefult to demo overfitting - you will have to fit a complex model to it

""""
This dataset is intentionally designed to demonstrate:

nonlinear classification
decision boundaries
overfitting
underfitting
feature engineering
kernel methods

It’s one of the most important toy datasets in machine learning education.
"""

# X -> 300 training examples, 2 features per example
# so a 2D array of 300 entries where each entry is a 1D vector of 2 feature values

# NOTE: out of the 300 tr. eg., 150 correspond to one moon cluster data points and have associated binary 
# outcome of 0; the other 150 corresponds to the second moon cluster data points that have associated
# binary outcome of 1

# noise is also added

# random_state=42, every run, you get the same moon data points (for both moons corresponding to the 
# 2 binary outcomes)

plt.scatter(X[:,0], X[:,1], c=y)
plt.show()