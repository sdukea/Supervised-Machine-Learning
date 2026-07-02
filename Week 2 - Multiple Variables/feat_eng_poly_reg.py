"""
Goals
In this lab you will:
- explore feature engineering and polynomial regression which allows you to use the machinery of linear regression to fit very complicated, even very non-linear functions.
"""

import numpy as np
import matplotlib.pyplot as plt
# from lab_utils_multi import zscore_normalize_features, run_gradient_descent_feng

np.set_printoptions(precision=2)

# NOTE some great understanding:
# a model: your guessing machine
# - its a function with adjustable parameters in it
# - when you find the best parameters that the model will have (after training/G.D./other optimization)
# - then your model 'predicts' and returns a prediction; returns y-hat for each training example

# in U.L.R.:
# - your model is – wx + b
# - and when you find the right w and b that minimize the cost function J(w, b) (w not a vector),
# - then your model returns prediction/y-hat

# NOTE: a pattern in regression is how y (actual label) changes as the input features change
# NOTE: there are three 'y's you have to know:
# 1. y = f(x)
# eg: y = 1 + x^2
# this is the ground truth function; this is essentially a function that tells you how the
# input features you have (like X_train) are related to the corresponding ground truth labels you
# have (like y_train)
# this is usually/in most cases unknown because you just get the actual labels only (with features as 
# well that correspond) and determining a relationship is totally undoable right off the bat

# the ground truth function is a rule that applies to any valid input – not just the training data
# works for all unseen and seen/trained/known inputs as well and is not just restricted to the ones
# in your training dataset

# Say you had this ground truth function: y =x^2

# x	 y
# 1	 1
# 2	 4
# 3	 9
# 4	 16
# 5	 25

# Now, if I asked what y is when x = 10,
# -> x = 10 is not in the training set
# -> but the rule can still tell the answer 
# = (10)^2 = 100
# The rule was NEVER LIMITED TO THE FIVE POINTS

# These data points are only a small part of the inputs –
# there are INFINITELY MANY INPUTS

# Now, say you only saw:

# x	 y
# 1	 1
# 2	 4
# 3	 9
# 4	 16
# 5	 25

# Without telling you the rule, someone asks 
# what is y when x = 10?

# No - we will have to build a model for that

# A model that will approximate to this ground truth function that exists not here
# but EXISTS

# The GTF 'knows' what happens for every possible input

# Say you have a 2 hour movie:
# You take screenshots at 10 minutes, 30 minutes, 60 minutes and 90 minutes

# This is your dataset; the entire movie is the ground truth function

# Just because you only have four screenshots doesn't mean the movie only exists 
# at those four moments.

# It still exists at 103 minutes, 11 minutes and the 31st minute, even though you never

# captured them.

# That's why we say:
# the dataset is actually a finite sample from that underlying function
# what does this mean:
# 1. there are infinitely many inputs i.e. feature values/vectors
# 2. in regression, feature values are all real-values and hence there are infintely many of them
# 3. so out of the many possible feature values/vectors there exists, you only have a finite set in
# training; that's why when you have features values (just an eg.) like 1, 2, 3, 5, 6, 7, 9, 11 
# and their corresponding labels you'd think why don't we have the numbers in between 
# (a doubt you got eariler) it's because the space of possible inputs is mathematically infinite, 
# not just practically large. if you'd get all numbers from 1 to 100 as well, you'd still not have 
# the feat. values because you should've also got 1.1, 1.2, 1.3 and if so, you should've also got
# 1.05, 1.025, 1.0175 and it just keeps going on – its infinite values for features.
# that's why the data set is alwaya a finite sample
# |
# in your dataset, you might think that for every feature value/vector x, you apply the ground truth
# function and you get the associated true label 'y';
# but in reality, this is only partially true.
# the 1. equation, in practice, will actually be:
# y = f(x) + ϵ
# where ϵ is the noise or everything affecting y – your true output label – that is not captured
# by your input x (any of them)

# NOTE: The ground truth function is reality itself, viewed only through the features you decided 
# to measure - and hence our feature/predictors/dataset is incomplete (and inconsistent)

# say x = hours studied (just one feature) and y = exam score
# you might think y = f(x) (some ground truth function that we assume is universal)
# but in reality, exam score is also affected by sleep quality, stress, luck, question difficulty
# and so forth, which isn't captured by our 'x'
# so what happens is this: for the same 'x'
# study 5 hours –> score 78
# study 5 hours -> score 83
# study 5 hours -> score 80
# our 'x' stays the same but in the real world, 'x' and ϵ – the features that our 'x' does not capture
# – affect 'y' the score
# the scores are different because even though 'x' stays unchanged, 'x' and some other feature that we
# don't know and haven't accounted for is changing and that affects the score
# and this other, unaccounted for features are given by ϵ – the noise
# |
# so if y = f(x) + ϵ in the real world case, how is y = f(x) even relevant to talk about?
# think of f(x) – the ground truth universal function – as the average or expected outcome for a 
# given x; formally given by y = f(x) = E[y∣x] 
# and our model y-hat tries to learn/approximate to f(x) and not f(x) + ϵ because
# ϵ is random and cannot be learned; we know it will exist and bother us no matter what but it cannot
# be learned in any way because it keeps on changing as there are infinitely many possible features/
# variables that will affect 'y' and it is impossible to grab hold of all these variables no matter what
# takeaway: the world is not perfectly predictable, your features don't capture everything, so outcome/ground truth 
# (y) = signal (f(x)) + noise (ϵ) and model can only learn from signal
# so y = f(x) + ϵ is the real world equation; when you collect data and you have features with associated
# true output labels
# y = f(x) -> the systematic part; what's predictable from 'x'
# ϵ -> random part; not predictable by any means as there are infinitely many possible variables that
# could undergo change and affect 'y' and we cannot account to grab hold of all these variables to 
# estimate 'y'
# so y = f(x) is relevant because it is the only part you can ever hope to learn
# you can't learn ϵ:
# - it's random
# - changes every time
# - not tied to 'x' in a consistent way (if so, it would be part of f(x) – the function)
#   but still affects 'y'
#   knowing 'x' does NOT let you predict ϵ in any way and its totally random
#   if you know x = 5, you cannot predict if ϵ – totally existent and bothers us no matter what – will
#   push y up or down
#   if you knew what ϵ is, then that's not noise anymore – it'll be part of the function
#   and the function knows how y will change when 'x' changes
# |
# let's talk about the output label 'y' now:
# - for the same feature value/vector x, you might have different output labels
#   so, when x = 5, y = 78, when x = 5 again (somewhere down the rows in data), y = 80 now and when
#   x = 5 again after some while down in the data, y = 83 now
#   so you understand this – ϵ affects what 'y' is because everything that affects 'y' but is not
#   predictable from x is what ϵ is (noise)
#   there are missing features/variables that are affecting y that we still haven't accounted for
#   and x alone isn't predictive/deterministic of what 'y' is
# - and this can totally happen – you will see for a single feature value/vector x, different set
#   of possible values for 'y' because of ϵ noise

# NOTE: To explain polynomial regression and the use of 'y' = 1 + x^2 here
# - we use 'y = 1 + x^2' as the ground truth function to 
# - produce/generate synthetic data to work with
# - i.e. output labels for a given input/variable x that we have
# - so we defined this universal rule just for the sake of demonstration
# - and we generate data from it
# - and we know what f(x) is exactly
# - so y = 1 + x^2 is literally the true underlying relationship for the 'x' we have
# - and the y we synthetically generated
# - actually, y = f(x) stays unknown but here
# - we don't have data of output labels and features x to be in a state where
# - f(x) should stay unknown and wherein we should actually model the relationship to generalize to new
# - unseen data (make a new model that learns the ground truth function and generalize well)
# - for demonstration purposes, we need some output labels and feature values 'x' because
# - any way, the data that we get from the real world will have a ground truth function relating
# - the features we have and the output labels associated (plus some ϵ)
# - so we keep this idea true; we don't have 'x' or 'y' that relate to each other right now
# - so why not create a rule that we can reverse-use to produce data of 'y' and we can create some
# - corresponding 'x' and yay – we have some data to use now (synthetic)
# - yes, to simulate real world data, y = 1 + x^2 + ϵ (noise) could be used but we don't want to do 
# - that now – we just want to create a noise-free world for now for easy demonstration and understanding
# NOTE: This y = 1 + x^2 is the ground truth WE CREATED to reverse-create data set i.e.
# - the output labels y that we created for some 'x' were synthetically reverse-created with our
# - pre-made ground truth; NOTE: but still a ground truth -> actual rule that produced data we have
# - regardless the way we did that (reverse-created or inferred/realized/approximated to from data we got
# first of features and output labels)
# - this way of defining a rule first and then reverse-creating the data set that abides to this
# - rule makes this ground truth function rule generalize-able to unseen inputs as well, unless and until
# - the future, unseen data (not in training set we had/created) follows the same underlying process or
# - distribution as the training data we had/created
# - because ground truth function is: the actual rule that generated the data you're working with
# - so, DATA first and then GROUND TRUTH is adapted/realized/approximated to
# - here, its GROUND TRUTH definition first and then DATA creation using it next
# - either way, this function relates to the data set you're working with (some 'x' and created
# - 'y' using this ground truth - the reverse way) and so we can still call this ground truth even 
# - though we've taken the reverse way
# - even though you might think this is the ground truth function for this simulation/demonstration and
# - not generalizable to new data, this is not the case
# - this new rule will still generalize to new unseen data wherein this unseen data should have a 
# - similar distribution to that of the training data we had/created and which the ground truth function 
# - relates to (the 'x' we had/created and 'y' we had/created)


# so, ground truth – continous, universal rule that exists for all seen/unseen input and what the output
# should be
# training data – a few observed points from it
# and the model that you train – y-hat = f_parameters(x) (or) model(x)
# will try to approximate to this ground truth function, but it may fail
# in reality, not enough data, noise and bad modelling will result in only close approximations#
# but the goal is always: y-hat (model) ≈ f(x)

# 2. y_train
# as mentioned just eariler in 1.
# this is the labels/values for each training example you hav 

# in M.L.R.:
# - your model is – w0x0 + w1x1 + w2x2 + ... + wnxn + b
# - and after finding the best possible parameters w0 to wn,
# - your model returns y-hat for each training example

# in P.L.R.
# – feature engineering changes what a linear model fits/can represent
# - the normal w1x (or wx) + b linear model has a
# 1. constant slope
# 2. straight line relationship
# 3. every increase in x changes y by the same amount

# but real world relationships are often curved
# eg. temp vs energy, speed vs fuel efficiency, population growth and so on
# |
# so instead of changing the learning algo. (linear regression), we just change the
# features

# we transform our original features 'x' into
# x^2, x^3, ....

# and you learn weights for these TRANSFORMED features

# so model becomes: y = w1x + w2x^2 + w3x^3 + ...

# why still linear regression: model is non-linear in x (powers increase and model is quadratic, cubic
# and so on) but linear in weights
# weights don't have power – they aren't multiplied together or exponentiated

# so, its actually polynomial linear regression

# higher degrees polyomials, more flexible shapes, more the ability to fit complex data, 
# and more danger of overfitting

# NOTE: huge feature value magnitudes as a result of squaring, cubing will create unstable gradients,
# wild predictions, and extreme sensitivity
# so that's why scaling, regularization and lower-degree models are usually preferred

# create data
x = np.arange(0, 20, 1)

# create ground truth function and get true labels for created 'x'
y = 1 + x**2
# NOTE: if ground truth is used to produce data (reverse way), then that function automatically
# becomes y-hat because parameters are considered already set and we assume it will generalize to new
# data as well (if it has the same distribution as training data we created ourselves using the g.t.f
# in our case here)

X = x.reshape(-1, 1)
# reshape into a 2D array
# so X has shape (4,1) – a 2D array with just 1 column and 4 rows
# 2D array is what traditional models accept and so even though our 'x' is 1D, we need that
# data in the form of 2D 
# will now look like:
# [[1],
#  [2],
#  [3],
#  ...]

# let's see why a plain linear model fails to fit non-linear data

# if you plot x and y, you see that you get a parabolic relationship between them

# and if you train linear regression:

"""
model_w,model_b = run_gradient_descent_feng(
    X, y,
    iterations=1000,
    alpha=1e-2
)
"""

# and the model here only knows how to fit a straight line i.e
# fitting to x that has a linear relationship with y

# but here, you have a non-linear, quadratic relationship between 'x' and 'y'

# and so a linear model built to fit linear relationships will not be able to fit our data with its
# quadratic relationship here:

plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("no feature engineering")

# plt.plot(x,X@model_w + model_b, label="Predicted Value")
# plt.xlabel("X"); plt.ylabel("y"); plt.legend(); plt.show()

# NOTE: X @ model_w -> matrix multiplication

plt.show()

# you see that x (X is 2D array of x) and y have a parabolic relation
# but when you develop a linear model to learn quadratically related data
# and after you learned the model to get model_w and model_b
# and use that to predict values for each of the 'x' you have (just using training data again – not an issue as
# we don't have a test set so for each tr. eg. we already have we're getting a prediction and plotting
# it and joining all points together –> by joining all predictions for each tr. data we have (showing you error
# as well evidently) we see that there are infinitely many possible feature values and predictions associated
# because the graph, when you keep zooming in and seeing even more microscopically, it just keeps going on)

# you see there is not a great fit – a straight line model for parabolic data
# and there is high error – distance between actual values and predicted ones

# so, what do you do:

# you engineer features

x = np.arange(0, 20, 1)

y = 1 + x^2

# same thing again

X = x**2

X = X.reshape(-1, 1)

"""
model_w,model_b = run_gradient_descent_feng(X, y, iterations=10000, alpha = 1e-5)
"""

"""
plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("Added x**2 feature")
plt.plot(x, np.dot(X,model_w) + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()
"""

# NOTE: np.dot() can also be used for matrix multiplication

# now when you plot predictions for each training data/example, its a near perfect fit

# its still linear regression only – we have linear weights till now
# model_w are all linear values, not exponentiated

# its just that we engineer features

# ––––

# now in the above example, we know that x^2 was required.
# sometimes it may not always be obvious as to which features are required i.e. which exponentiation
# will be required
# we could do: adding in potential features that are exponentiated and finding the most useful

x = np.arange(0, 20, 1)

y = x**2

# let this be our ground truth

X = np.c_[x, x**2, x**3]

# c_ -> concatenate arrays column wise

# so if x = [1 2 3]
"""
X = [
     [1, 1,  1],
     [2, 4,  8],
     [3, 9, 27]
]
"""

# each column is what np.c_ gets in '[]'
# X is still a 2D array

"""
model_w,model_b = run_gradient_descent_feng(X, y, iterations=10000, alpha=1e-7)

plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("x, x**2, x**3 features")
plt.plot(x, X@model_w + model_b, label="Predicted Value")
plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()
"""

# now if you did the above to scatter 'x' and 'y' with a parabolic relation and then plot x (tr. set but we
# use it for evaluation) and model prediction trained on engineered features x, x^2 and x^3 to get model_w – a 
# vector of three values for each feature (x, x^2 and x^3) – and model_b

# you see that the model will be something like:
# 0.08x + 0.54x^2 + 0.03x^3 + 0.0106
# ^ w1    ^ w2      ^ w3

# by getting the weights for each feature, as you know:

# the higher the weight associated for a feature, the larger the importance of that feature
# because higher w, small change in x, large change in y, high sensitivity and great feature to look at
# and so gradient descent offers us a way to see which features are emphasized by the weights they are associated
# with
# and in our case, the most useful feature is x^2 as it has the highest weight
# NOTE: running G.D. for a longer time will show that the weights associated for x and x^3 will reduce and hence
# the impact of these features will be lowered and not needed and more importance will be given for x^2 and
# the weight will increase for this

# NOTE: the scale/range of features here should be same or else you wouldn't be able to know the exact
# magnitude of weights and cannot infer

# an alternative view

# create target data
x = np.arange(0, 20, 1)
y = x**2

# engineer features .
X = np.c_[x, x**2, x**3]   #<-- added engineered feature
X_features = ['x','x^2','x^3']

fig,ax=plt.subplots(1, 3, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X[:,i],y)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("y")
plt.show()

# note that if the relationship between our features and target is a quadratic, SQUARING the features seems to
# work.
# and if features are squared, like X = x**2 (x^2), and the ground truth function is also a parabolic function,
# the features and the output labels (via the g.t.f.) will be linearly relative – both feature and output labels
# are raised by a power of 2.
# And hence, 'linear' regression will work best here, by polynomial-izing the features to square to match the
# g.t.f. (that imposes a square/quadratic relationship) so as to make both the same power and hence linear (think
# as if powers cancel and will be made linear)

# that's why the graph of X = x**2; the engineered feature to square; and y; that imposes a square/quadratic 
# relation; that has been plotted above will have a linear relationship

# and hence, modelling with a linear model now will return model_w and model_b (w and b parameters) that are
# much more effective and building a model now is just trying to fit a linear model which w and b we got will
# do their best on now

# So poly. reg is feature engineering and just ordinary linear regression after

# scaling features

# as said earlier, scaling features is really important as it makes g.d. run much faster plus getting associated
# weights for each engineered feature would be clearly evident of information

# create target data
x = np.arange(0,20,1)
X = np.c_[x, x**2, x**3]
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X,axis=0)}")

# add mean_normalization 
X = zscore_normalize_features(X) # type: ignore
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X,axis=0)}")

# z-score normalize it!

# now g.d. will run much faster

x = np.arange(0,20,1)
y = x**2

X = np.c_[x, x**2, x**3]
X = zscore_normalize_features(X) # type: ignore

model_w, model_b = run_gradient_descent_feng(X, y, iterations=100000, alpha=1e-1) # type: ignore
# more aggressive value of alpha = learning rate

plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("Normalized x x**2, x**3 feature")
plt.plot(x,X@model_w + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()

# complex functions can also be modelled

x = np.arange(0,20,1)
y = np.cos(x/2)

X = np.c_[x, x**2, x**3,x**4, x**5, x**6, x**7, x**8, x**9, x**10, x**11, x**12, x**13]
X = zscore_normalize_features(X) # type: ignore

model_w,model_b = run_gradient_descent_feng(X, y, iterations=1000000, alpha = 1e-1) # type: ignore

plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("Normalized x x**2, x**3 feature")
plt.plot(x,X@model_w + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()

# NOTE:
# relationship between x and y = x^2 --> non-linear/quadratic
# so we engineer features ot make features and trg

# FINAL UNDERSTANDING of ground truth function and approximation

# say you're predicting house price using size of house

# y = f(x) is the ground truth; 
# - what this shows is that for THIS size of house, THIS is the house price for any size of house
# (and inherent house price)
# so there exists some underlying mechanism/process in nature or reality that relates to the size
# of house (and price of house)
# this y = f(x) does not necessarily mean it is something
# 1. that is deterministic
# 2. humans should already know
# 3. simple
# 4. with a neat equation

# y = f(x) can be explained by physics, biology, economics

# we know that these patterns exist - for some size of house x, there exists THIS right price y
# and it is determined by physics, economics or nature sciences - but we won't necessarily know the
# exact equation

# for ANY valid input x, reality defines some output behavior.
# The function exists over the ENTIRE input domain.
# The true function defines outputs everywhere i.e. for every single input

# so if you had 2 features and associated labels y,
# the ground truth function is y = f(x1, x2) and is defined for the entire possible 
# space of those two features (infinte real values for these features exist)
# so, for any valid combination of (x1, x2), reality defines some output behaviour

# NOTE: also, yes, this ground-truth function y = f(x1, x2) from data of 2 features and associated labels
# y also generalizes to new, unseen data because the unseen data is also in the same distribution as 
# x1 and x2 - just not captured by the real world dataset we have 

# and our dataset is only a small subset of the ground truth function i.e.
# only a small subset of the infinite, real-valued feature values that we're using for the model
# and the ground truth function considered for these n features f(x1...xn) and the infinte values
# for each feature
# why (let's see on the basis of input features and the associated labels y):

# 1. input features
# we cannot collect infinite data
# - you cannot collect every size, every location, every year, every market condition and so on
# - you can maybe collect a million examples but you're data will always
# be TINY compared to reality's FULL space

# the input data you collect is also inconsistent
# - all input data always has
# a. measurement errors
# b. sensor inaccuracies
# c. randomness
# d. hidden variables
# e. human inconsistency
# (you get the point)

# the input data does not capture all relevant features in reality
# - this adds to point 1

# 2. the output labels (in data)
# these output labels are also associated with the inconsistent, finite input features x
# and so these output labels are also considered only a small subset of reality/infinite feature space
# in our dataset, these output labels are OBSERVED OUTPUT labels which are actually 
# equal to y = f(x) + e
# where 'e' is the noise from the dataset as we've seen in '1. input features'
# and these observed labels are called actual, true, ground-truth labels in practical ML because they are
# the target values we train against
# |
# but yes, looking from a statistical point,
# these ground-truth, true, actual, output, observed labels are NOT always perfectly related to y = f(x)
# (that 'y' which is the ground-truth output of the ground-truth function)
# because in dataset, these actual, true, observed labels are actuall noisy, inconsistent and are taken
# as output wherein there are hidde variables, that aren't captured by the input features we have,
# affecting this output

# In practice, “true labels” usually means the observed labels in the dataset — even though statistically 
# they may be noisy realizations of a deeper underlying process. 

# we usually refer the labels provided in the dataset as our 'true output' labels
# by convention
# because we still need some kind of target, even if they're imperfect
# these labels are treated as authoritative supervision because in S.L., we need some target
# to learn from

# We may never know the exact underlying clean function/process because we only observe finite, noisy, 
# incomplete samples from reality, and the observed outputs already include randomness, hidden-variable 
# effects, and uncertainty.

# So y = f(x) is the function that predicts the true, reality output y for ANY valid input x (
# infinite, real-valued)

# you may never know what y = f(x) but we try to approximate to this, with our incomplete evidence
# (dataset) - that's what ML is


# –––––– REVISE

# the ground truth is the real world that generate the dataset
#                                                         |
#                                                        \/
# some number of features n and feature values that we derive/observe from the real world
# that we collect as data                                       |
#                                                              \/
# observation has two flaws; one is that we can NEVER observe/record ALL possible changes/features 
# for getting the outputs from - this is humanely impossible. This is why we have x/x_vec + noise
# where noise is everything that affected our output that was not in the dataset; the second thing
# is that we are inconsistent - we make mistakes/errors while collecting data so we NEVER truly
# interpret or see the ground truth output that the real world gave - so our y that we derived from
# features (that are both inconsistent and incomplete) is NEVER the actual ground truth - we rather
# call them the observed labels and still use them
#                                          |
#                                         \/
# we use these observed labels as our ground truth but statiscally, they are not the ground truth. 
# Inherently, they are y = f(x) + e where 'e' is the noise - everyhing that is not in our dataset 
# (as our feature/predictors are incomplete/inconsistent - we can never capture just f(x) without
# noise)    