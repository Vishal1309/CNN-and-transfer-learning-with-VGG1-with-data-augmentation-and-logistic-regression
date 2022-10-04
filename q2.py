import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import time
np.random.seed(42)

print("Varying M with N fix")
N = 5
M = np.arange(1000, 40000, 2000)
M_arr = []

train_time_arr = []
test_time_arr = []

for i in range(len(M)):

    X = pd.DataFrame(np.random.randn(N, M[i]))
    y = pd.Series(np.random.choice([1, 2], size=N))
    if(len(np.unique(y)) == 1):
        continue
    M_arr.append(M[i])
    logistic_model = LogisticRegression()
    # print(y)
    start_time = time.time()
    logistic_model.fit(X, y)
    end_time = time.time()
    train_time = end_time-start_time

    start_time = time.time()
    y_hat = logistic_model.predict(X)
    end_time = time.time()
    test_time = end_time-start_time

    train_time_arr.append(train_time)
    test_time_arr.append(test_time)

fig = plt.figure()
plt.plot(M_arr, train_time_arr, label="train_time")
plt.plot(M_arr, test_time_arr, label="test_time")
plt.title('Varying M with N fixed')
plt.xlabel("M")
plt.ylabel("Time in seconds")
plt.legend()
fig.savefig('Time_vs_M.png')

print('########################################################################')


print("Varying N with M fix")

N = np.arange(1000, 40000, 2000)
M = 5
N_arr = []

train_time_arr = []
test_time_arr = []

for i in range(len(N)):

    X = pd.DataFrame(np.random.randn(N[i], M))
    y = pd.Series(np.random.choice([1, 2], size=N[i]))
    if(len(np.unique(y)) == 1):
        continue
    N_arr.append(N[i])
    logistic_model = LogisticRegression()

    start_time = time.time()
    logistic_model.fit(X, y)
    end_time = time.time()
    train_time = end_time-start_time

    start_time = time.time()
    y_hat = logistic_model.predict(X)
    end_time = time.time()
    test_time = end_time-start_time

    train_time_arr.append(train_time)
    test_time_arr.append(test_time)
#######################################################

fig = plt.figure()
plt.plot(N_arr, train_time_arr, label="train_time")
plt.plot(N_arr, test_time_arr, label="test_time")
plt.title('Varying N with M fixed')
plt.xlabel("N")
plt.ylabel("Time in seconds")
plt.legend()
fig.savefig('Time_vs_N.png')

print('########################################################################')
