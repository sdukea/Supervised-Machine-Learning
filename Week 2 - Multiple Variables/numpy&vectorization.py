# learn all about NumPy, its application on vectorizaton and all things related

import numpy as np
import time


# vectors
# - ordered array of numbers
# - lower case bold letters
# - elements in vector are all the same type; does not contain characters and numbers ATST
#   a 2D vector has a dimension of 2 i.e 2 values in the vector (not a matrix with rows AND columns)
#   a 3D vector has a dimension of 3 i.e. 3 values in the vector
#   a nD vector has n dimensions i.e n values inside the vector
# - indexing a vector can be done; from 0 to n-1 in code; 1 to n normally
# - also, this is what RANK means NOTE: In NumPy:
#   rank 1: np.array([1, 2, 3])
#   rank 2: np.array([[1,2], [3,4]])
#   rank 3: np.array([[[1],[2]], [[3],[4]]])
#   its the number of '[]' you have when creating an array - the rank
# - dimension also means: the number of features/rows (.shape[0] as well; yields no: of rows/feat.)

# numpy arrays
# - numpy's basic data structure
# - is an indexable, n-dimensional array
# - containing elements of the same datatype
# - NOTE: In numpy, DIMENSION refers to the number of indexes of an array
#   a 1-D array has one index
#   - x = np.array([1.0, 2.0, 3.0])
#   - x's shape -> (3,)
#   - and you index 1-D arrays only with one set of '[]'
#   - and hence, dimension is only one because there is only one index (one set of '[]'/index)
#   for a 2-D array like
#   - x = np.array([[1.0, 2.0], [3.0, 4.0]])
#   - the shape is -> (2,2)
#   - because x is actually
#   - [1.0, 2.0]
#   - [3.0, 4.0]
#   - and you index with 2 index '[]'
#   - x[1][0] - indexing the first element out of the second element in the 2-D array
#   - and the same for 3-D array -> 3 '[]' and in NumPy, the number of indexes is dimension
#   - NOTE: In math, DIMENSIONS are the independent directions you can take
#   - So, [1.0, 2.0, 3.0] lives in a 3-D space
#   - There is no 'dimension' of a matrix - you say its rows x columns
#   - For a matrix, you can maybe tell the dimension of the rows because rows itself is a vector
#   - i.e. a row vector (and maybe tell the dimension of its column/column vectors as well)
#   - NOTE: In math, RANK is the number of linearly independent rows/columns
#   - For:
#   - [1.0  2.0]
#   - [3.0  4.0]
#   - the rank is 2 because there are two independent rows or columns IN A MATRIX
#   - The above definition for NumPy actually mirrors the idea that rank applies for a matrix because
#   - having multiple '[]' actually lets you build a matrix (think and imagine)
#   - If one row can be made from others, its redundant - does not contribute to rank.
#   - rank is basically how many unique directions your data actually spans
#   - [1.0  2.0]
#   - [3.0  4.0]
#   - Neither of the rows depend on each other - so they're independen
#   - and the rank of this matrix is 2
#   - But when it comes to this:
#   - [1.0  2.0]
#   - [2.0  4.0]
#   - Row 2 is 2 x row 1
#   - and row 2 is dependent on row 1 - rank is only 1 here as row 2 is redundant

# vector creation

a1 = np.zeros(4)
# same as
a2 = np.zeros((4,)) # (4,) 1-D array with 4 elements in it

# (4,2); creates a 2D array with 4 rows and 2 columns
# same as doing (but all with zeros): 

a2 = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8]
])

# you have 2 sets of '[]' making up this array inherently and so its 2 dimensional -> tuple as argument
# given also has 2 values -> the tuple is the 'shape' argument

# (4,2,3); creates a 3D array as you know how it looks (4 blocks; each block has 2 rows and 3 columns)

# again, same as doing (but all with zeros):

a2 = x = np.array([
    [  # block 1
        [1, 2, 3],
        [4, 5, 6]
    ],
    [  # block 2
        [7, 8, 9],
        [10, 11, 12]
    ],
    [  # block 3
        [13, 14, 15],
        [16, 17, 18]
    ],
    [  # block 4
        [19, 20, 21],
        [22, 23, 24]
    ]
])


# and so on; further blocking (a block's block's block's block('s) and so on) until 
# we catch its rows and columns and continue

# again, you have three sets of '[]' (color purple-pink is one, the blue is the second and
# the yellow is the third one - hence its 3 dimensional -> you also gave the tuple with 3 values
# -> the tuple is the 'shape' argument)

a3 = np.random.random_sample(4)

# all these creates a vector/array of length 4

# result: [0.0  0.0   0.0   0.0] for a1 and a2; all zeros
# result: [0.1  0.91  0.23  0.32] for a3
# creates an array of length 4 and each entry is a random number between 0 and 1

# (some more)
# NOTE: These take the 'shape' argument - the values inside - as seperate arguments and not 
# inside a tuple

a4 = np.arange(4)

# creates an array containing numbers from 0 to 3 (4 numbers)
# so: [0.0  1.0   2.0   3.0]

# .arange(start, stop, step)

a4 = np.arange(2, 6, 2)

# [2.0   4.0    6.0]


# manually specifying values

a5 = np.array([1, 2, 3, 4, 5])
a6 = np.array([6., 7, 8, 9, 10]) # *

# * all elements in an array should have the same datatype
# because '6.' is a float, it upcasts all others to the safest common type - float

# NOTE: there are 1-D arrays (rank 1)

print(a6.dtype)
# and so all become float and type of a6 is float64

# operations on vectors

# - elements of vectors can be accessed via: indexing and slicing
# - indexing: referring to an element of an array by its position
# - slicing: getting a subset of elements from an array based on indexes/indices

# indexing - 1-D array/vector

a7 = np.arange(10)

# shape of a7: (10,) - a 1-D array/vector

# each value inside a 1-D array is a scalar value so

# a7[2].shape will equal () - an empty shape (0 dimensional)

print(a7[-1])

# gives last element; you know it

# indexes should be within the range of the vector - or else error

# if for higher dimensions, say n, you have to give in n indexes to get that element

# either like: a7[index,index,index,index,...] or a7[index][index][index]...

# slicing

a8 = np.arange(10)

# a8[start:stop:step]

#access 5 consecutive elements
c = a8[2:7:1];     print("a[2:7:1] = ", c)

# access 3 elements separated by two 
c = a8[2:7:2];     print("a[2:7:2] = ", c)

# access all elements index 3 and above
c = a8[3:];        print("a[3:]    = ", c)

# access all elements below index 3
c = a8[:3];        print("a[:3]    = ", c)

# access all elements
c = a8[:];         print("a[:]     = ", c)

# single vector operations

a9 = np.array([1, 2, 3, 4])

print(f"a             : {a9}")

# negate elements of a
b = -a9 
print(f"b = -a        : {b}")

# sum all elements of a, returns a scalar
b = np.sum(a9) 
print(f"b = np.sum(a) : {b}")

b = np.mean(a9)
print(f"b = np.mean(a): {b}")

b = a9**2
print(f"b = a**2      : {b}")

# vector vector element wise operations 

# NOTE: vector -> 1-D array (you know it already)

x = np.array([1, 2, 3, 4])
y = np.array([-1, -2, 3, 4])

print(x + y)

# adds each element of x and y together orderly

# so: 1+ (-1), 2 + (-2), 3 + 3, 4 + 4

# if both arrays are not of the same shape, then you cannot perform operations; you'd get value error.
# cannot broadcast together two arrays with shape (4,) and (5,)

# scalar vector operations

z = np.array([1, 2, 3, 4])

b = 5 * z

print(b)

# yes, each element of z is multiplied with 5; scalar-vector operation in action

# vector vector dot product

def my_dot(a, b):

    # a: input vector
    # b: input vector with same dimension as a (matching dimensions; if not ValueError)

    # returns: x (scalar)

    x = 0

    for i in range(a.shape[0]):
        # NOTE: you know that .shape gives you rows and columns i.e .shape[0] gives you rows
        # in a 1-D array/vector, .shape is just (number of elements,)
        # and .shape[0] gives you the NUMBER of elements (which is essentially the number of
        # columns but you're fine - just get the fact that they give you the number of elements
        # and we're iterating over each element)

        x = x + a[i] * b[i]

    return x

# exactly like vectorization we saw; this method is OK and isn't as fast as using .dot(a, b)

a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

print(my_dot(a, b)) # answer: 70 (correct)

# can also do:

c = np.dot(a, b)

print(c) # same answer: 70

# need for speed: vector vs for loop

# let's see if vectorization (using np.dot) or the for loop is faster

np.random.seed(1) # every time you run generate random arrays, the set of randomized array values
# will now be the same if the seed is set
# any number is given inside; 42 is used because of a joke/reference
# 1 will have a random sequence that is maintained and won't change for every run
# 345913 will have a different random sequence but it is maintained and will not change for
# every run

a = np.random.rand(10000000)  
b = np.random.rand(10000000)

tic = time.time()  # capture start time
c = np.dot(a, b)
toc = time.time()  # capture end time

print(f"np.dot(a, b) =  {c:.4f}")
print(f"Vectorized version duration: {1000*(toc-tic):.4f} ms ")

tic = time.time()  # capture start time
c = my_dot(a,b)
toc = time.time()  # capture end time

print(f"my_dot(a, b) =  {c:.4f}")
print(f"loop version duration: {1000*(toc-tic):.4f} ms ")

del(a);del(b)  #remove these big arrays from memory

# vectorized duration: 14 ms
# for loop duration: 2628 ms

# so use vectorized method - np.dot

# vector vector operations on course 1

# training examples will be stored in X_train of shape (m,n)
# - m -> number of training examples/rows
# - n -> number of features/columns
# - its a 2-D array/matrix

# - w -> a 1-D vector/array with shape (n,)
#   represents our weights for each feature so
#   has to be the size of the number of features (one w for one feature)
#   so for n features, n weights and n weights only and that's it (1-D array is enough)
# - will be looping through examples by indexing (i) to extract each example to work on individually
# - X[i] will return a value of shape (n,) because
#   as X is a 2-D array, each element in the array is a 1-D array (as 2-D arrays are made up of
#   1-D arrays and the shape of the 1-D array was its columns essentially as we saw before and
#   the number of columns represents the feature(s) n we have and so that's why the shape is
#   (n,).)
# - operations involving X[i] are often vector-vector

# eg:

X = np.array([[1], [2], [3], [4]])

# actually like:

X = np.array([
    [1],
    [2],
    [3],
    [4]
])

# single feature here, obviously and
# 2D as we have two sets of '[]' - a purple-pink and blue one
# and shape is (m/rows=4 and n/columns=1)

# NOTE: for nD array, we will have n different values in the 'shape' tuple (...n)

print(X.shape) # (4,1)

print(X[1].shape) # (number of features,) = (1,)
# as explained above

# matrices

# - are 2-D arrays
# - elements of a matrix are of the same type
# - denoted with capital bolded letters (we just capitalize in code)
# - we index them with a two dimensional index

# - we use 2D matrices to hold data
# - consisting of m tr. eg and n features and so
# - X makes up a size of (m rows and n columns); exactly like how we know data looks like
# - (rows as tr. eg and columns as seperate features)

# matrix creation

a = np.zeros((1, 5))                                       
print(f"a shape = {a.shape}, a = {a}")                     

a = np.zeros((2, 1))                                                                   
print(f"a shape = {a.shape}, a = {a}") 

a = np.random.random_sample((1, 1))  
print(f"a shape = {a.shape}, a = {a}") 

# all these, as you already know, are creation of arrays (vectors/matrices) that take the
# shape as a tuple

a = np.array([[5], [4], [3]]);   print(f" a shape = {a.shape}, np.array: a = {a}")
a = np.array([[5],   
              [4],   
              [3]]); 
print(f" a shape = {a.shape}, np.array: a = {a}")

# you can create them seperately as well, as seen above

# indexing

a = np.arange(6).reshape(-1, 2) # create a matrix easily!

# .reshape changes how your data is organized, without changing the data itself

# so, you have: np.arange(6)

# [0 1 2 3 4 5], and the shape is (6,)

# now, you can reshape it into:

# 1. (2,3) - makes it have 2 rows and 3 columns

# [0 1 2]
# [3 4 5]

# .
# .
# .

# and you get the point.

# NOTE: The to-reshape shape tuple that you give in as the argument should be of the same
# size as the array (matrix) that is getting reshaped

# (2,3) -> rows x columns = 6 elements in total
# exactly matches the number of elements we have in a

# (-1, 2) -> figure out the number of rows automatically but I need 2 columns regardless

# so, it first splits the data into two columns and then automatically arranges the rows so that it
# accomodates correctly to our number of elements

# so, 'a' is actually

# [0 1]
# [2 3]
# [4 5]  # 2 columns and then rows are split automatically based on number of elements we have so
         # that we accomodate all values in the 2 columns that is pre-set and how many so ever number
         # of rows it takes: 6 / 2 = 3 rows

# and you know how to index:

# you specify either two '[]' each with an index (that does not exceed)
 
# or

# you specify one '[]' and you enter two values each corresponding to the index (that does not 
# exceed) - FASTER

# slicing

a = np.arange(20).reshape(-1, 10)

# split into 10 and figure out rows automatically such that we accomodate all 
# values we have (20 of them)

a[0, 2:7:1]

# result: [2 3 4 5 6]

# row 0/first item i.e. the first row in our case

# columns/values from 2 to 6 with step = 1 (normal step)
# - you are only able to do this because the next item that we're indexing further into the
# matrix after item 0 our first index are all just values; item 0/row 0 is a 1-D array and inside the
# 1-D array which our next index corresponds to is all just 0-D arrays or values and we can put an 
# index to those values like 2:7:1 and get specific values out of it

a[:, 2:7:1]

# from all rows, and 2:7:1 index

# result:

# [
#  [ 2  3  4  5  6]
#  [12 13 14 15 16]
# ]

a[:,:]

# all elements

a[1,:]

# all row 1 elements

a[:,1]

# all columns 1 elements