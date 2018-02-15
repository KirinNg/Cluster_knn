import numpy as np
import time

def getsame(a,b):
    c = a+b
    one = np.sum(c==1)
    two = np.sum(c==2)
    if (one+two)!=0:
        loss = two*1.0/(one+two)
        return loss
    else:
        return 0

def getdif(a,b):
    c = a+b
    one = np.sum(c==1)
    two = np.sum(c==2)
    loss = two*1.0/(one+two)
    if (one+two)!=0:
        loss = two*1.0/(one+two)
        return 1-loss
    else:
        return 0

data = np.load("data_1c.npy")
Matrix = np.zeros([data.shape[0],data.shape[0]])
for i in range(Matrix.shape[0]):
    for j in range(Matrix.shape[1]):
        Matrix[i][j] = getsame(data[i],data[j])
    print(i/Matrix.shape[0]*100)
np.save("SamMatrix_1c500.npy",Matrix)

data = np.load("data_2c.npy")
Matrix = np.zeros([data.shape[0],data.shape[0]])
for i in range(Matrix.shape[0]):
    for j in range(Matrix.shape[1]):
        Matrix[i][j] = getsame(data[i],data[j])
    print(i/Matrix.shape[0]*100)
np.save("SamMatrix_2c500.npy",Matrix)
