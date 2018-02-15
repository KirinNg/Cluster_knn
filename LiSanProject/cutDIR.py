import numpy as np
import os

DIR = list(np.load("DIR.npy"))     #48970
'''
LIST_ALL = list(np.load("LIST_1.npy"))+list(np.load("LIST_2.npy"))

cal = np.zeros([len(DIR)])

t = '%/\#$&|{}~'

for i in range(len(LIST_ALL)):
    inside = open(LIST_ALL[i]).read()
    for k in t:
        inside = inside.replace(k,' ')
    inside = inside.split(" ")
    for j in range(len(inside)):
        if inside[j] in DIR:
           cal[DIR.index(inside[j])] +=1
    print(i/len(LIST_ALL)*100)
max = np.max(cal)
np.save("cal_of_DIR.npy",cal)
'''
cal = np.load("cal_of_DIR.npy")
remove = []
for i in range(len(DIR)):
    if(cal[i]>100):
        remove.append(DIR[i])
for i in remove:
    DIR.remove(i)
np.save("DIR_cut100.npy",DIR)