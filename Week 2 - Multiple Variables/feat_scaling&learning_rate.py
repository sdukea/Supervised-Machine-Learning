"""
In this lab you will:
- Utilize  the multiple variables routines developed in the previous lab
- run Gradient Descent on a data set with multiple features
- explore the impact of the *learning rate alpha* on gradient descent
- improve performance of gradient descent by *feature scaling* using z-score normalization
"""

import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

X_features = ['size(sqft)','bedrooms','floors','age']

fig,ax=plt.subplots(1, 4, figsize=(12, 3), sharey=True)

# subplots - multiple plots in a single run
# 1 - 1 row
# 4 - 4 different plots on that 1 row
# so creates 4 empty graphs sitting next to each other
# fig - the outer box/container holding inner graphs
# ax - [plot1, plot2, plot3, plot4]

# figsize - wide (12) and short (3)
# sharey - all plots use the same y axis -> price



for i in range(len(ax)): # for each plot
    ax[i].scatter(X_train[:,i],y_train) # plot the feature - the index [:, i] and y_train
    ax[i].set_xlabel(X_features[i]) # set xlabel
ax[0].set_ylabel("Price (1000's)") # you just have to set the label for one plot - all plots share it
plt.show()

# you can now see each feature being plotted with the target

# learning rate

# a lot of proprietary modules used to demonstrate choosing the best learning rate
# so, needed those but couldn't get them here so skip it - understand the intuition though

# feature scaling

# it is important to rescale features so that they have a similar range
# saw three methods:
# 1. feature scaling; dividing each feature value of that feature by its maximum value - good for when
# you have a positve feature (all feat. values are positive) --> xnew = x / max
# 0 to 1 scale (not centered in any way)
# |
# could also do: x - min / (max - min); applies for any feature
# 0 to 1 scale again (not centered in any way at like 0.5 or something)

# 2. mean normalization: x - mu / max - min
# centered around 0
# (x - mean) --> makes values above mean positive and values below mean negative
# (max - min) --> tells you how wide the feature is; dividing it by it scales the data based 
# on how spread out it is; you're saying shrink everything relative to how wide it is because dividing
# by its range shrinks the data more appropriately 

# 3. Z-score normalization:

# will see now:

def zscore_normalize_features(X):

    """
    computes  X, zcore normalized by column
    
    Args:
      X (ndarray (m,n))     : input data, m examples, n features
      
    Returns:
      X_norm (ndarray (m,n)): input normalized by column; the normalized dataset
      mu (ndarray (n,))     : mean of each feature
      sigma (ndarray (n,))  : standard deviation of each feature
    """
    mu = np.mean(X, axis=0)

    # axis -- denotes the number of directions/axis when you plot or see the data
    # axis=0 -- columns; calculate mean for each column/each feature
    # axis=1 -- rows; calculate mean for each row
    # that's it; axis=2 won't work as X is a 2D array and you only have 2 directions/axis

    # mu is a 1-D array having the n mean values for all n features in data X (2D data)

    sigma = np.std(X, axis=0) # same, for all feature/column values

    # sigma is also an array having n sigma values for all n features

    X_norm = (X - mu) / sigma

    # so, mu and sigma look like:
    """
    mu    = [μ₁, μ₂, μ₃, μ₄]
    sigma = [σ₁, σ₂, σ₃, σ₄]
    """

    return (X_norm, mu, sigma)

# --- Compute statistics ---
mu    = np.mean(X_train, axis=0)
sigma = np.std(X_train, axis=0)

# --- Transform data ---
X_mean = X_train - mu
X_norm = (X_train - mu) / sigma

# visualization

# --- Plot ---
fig, ax = plt.subplots(1, 3, figsize=(12, 3))

# 1. Unnormalized
ax[0].scatter(X_train[:, 0], X_train[:, 3])
ax[0].set_xlabel(X_features[0])
ax[0].set_ylabel(X_features[3])
ax[0].set_title("Unnormalized")
ax[0].axis('equal')

# 2. Mean-centered
ax[1].scatter(X_mean[:, 0], X_mean[:, 3])
ax[1].set_xlabel(X_features[0])
ax[1].set_ylabel(X_features[3])
ax[1].set_title("X - μ")
ax[1].axis('equal')

# 3. Z-score normalized
ax[2].scatter(X_norm[:, 0], X_norm[:, 3])
ax[2].set_xlabel(X_features[0])
ax[2].set_ylabel(X_features[3])
ax[2].set_title("Z-score normalized")
ax[2].axis('equal')

# --- Layout ---
plt.tight_layout()
fig.suptitle("Distribution of features before, during, after normalization")

plt.show()

# just plotting the before and after normalizing features and viewing their centres.

# normalize the original features
X_norm, X_mu, X_sigma = zscore_normalize_features(X_train)
print(f"X_mu = {X_mu}, \nX_sigma = {X_sigma}")
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")   
# you're printing the peak-to-peak value i.e. max - min of each column value/feature
# for X_train alone, it is different for each column: 1252 for feature 1, 3 for feature 2,
# 1 for feature 3 and 10 for feature 4
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")
# now, if you did max - min for every column value, you'd have them in the same range (almost very
# similar to each other)
# [2.45 2.41 2.12 2.45]

# a lot of other visualizations that follows
# all good - you're fine to move on
# refer intership project code files for more on this - very well expounded there!