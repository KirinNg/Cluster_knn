import numpy as np
import os

DIR = []

def dirlist(path, allfile):
    filelist =  os.listdir(path)
    for filename in filelist:
        filepath = os.path.join(path, filename)
        if os.path.isdir(filepath):
            dirlist(filepath, allfile)
        else:
            allfile.append(filepath)
    return allfile
  
list1 = dirlist("20news-part1", [])
list2 = dirlist("20news-part2", [])

#np.save("LIST_1.npy",np.array(list1))
#np.save("LIST_2.npy",np.array(list2))

t = '%/\#$&|{}~'

for i in range(len(list1)):
    all_text = open(list1[i]).read()
    for k in t:
        all_text = all_text.replace(k,' ')
    all_text = all_text.split(" ")
    DIR = DIR + all_text

for i in range(len(list2)):
    all_text = open(list2[i]).read()
    for k in t:
        all_text = all_text.replace(k,'')
    all_text = all_text.split(" ")
    DIR = DIR + all_text

print(len(DIR))
DIR = list(set(DIR))
del DIR[0]
print(len(DIR))
#np.save("DIR.npy",np.array(DIR))

