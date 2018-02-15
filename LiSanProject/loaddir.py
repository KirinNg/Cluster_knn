import numpy as np
import os

DIR = list(np.load("DIR.npy"))     #48970
LIST_ALL = list(np.load("LIST_1.npy"))

X = np.zeros([len(LIST_ALL),len(DIR)],int)
t = '%/\#$&|{}~'
num=0
for i in range(len(LIST_ALL)):
    inside = open(LIST_ALL[i]).read()
    for k in t:
        inside = inside.replace(k,' ')
    inside = inside.split(" ")
    for j in range(len(inside)):
        if inside[j] in DIR:
            X[i,DIR.index(inside[j])] = 1
        else:
            if(inside[j]!=''):
                num+=1
    print(i/len(LIST_ALL)*100)
print(num)

np.save("data_1.npy",X)

LIST_ALL = list(np.load("LIST_2.npy"))

X = np.zeros([len(LIST_ALL),len(DIR)],int)
t = '%/\#$&|{}~'
num=0
for i in range(len(LIST_ALL)):
    inside = open(LIST_ALL[i]).read()
    for k in t:
        inside = inside.replace(k,' ')
    inside = inside.split(" ")
    for j in range(len(inside)):
        if inside[j] in DIR:
            X[i,DIR.index(inside[j])] = 1
        else:
            if(inside[j]!=''):
                num+=1
    print(i/len(LIST_ALL)*100)
print(num)

np.save("data_2.npy",X)