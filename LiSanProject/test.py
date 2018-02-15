import matplotlib.pyplot as plt
import numpy as np
import os

data = np.load("result/result2.npy")
plt.imshow(data)
plt.show()
'''
file = open("tensor2.txt",'w')
for i in range(100):
    for j in range(100):
        w = str(i)+"\t"+str(j)+"\t"+str(data[i][j]*data[i][j]*100)+"\n"
        file.write(w)
        print(w)
file.close()
'''
f1 = plt.figure(1)
plt.subplot(211)
plt.scatter(range(data.shape[0]), np.sum(data,1))
plt.show()