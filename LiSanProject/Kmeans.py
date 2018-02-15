import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from numpy import linalg

def RBF(d,theta):
    d = np.exp(-np.square(d)/(2*theta*theta))
    return d

def K(data,k):
    k = data.shape[0]-k
    for i in range(data.shape[0]):
        sort = np.argsort(data[i,:])
        for j in range(k):
            data[i,sort[j]] = 0
    new = (data+data.T)*0.5
    return new

def getNormLaplacian(W):
    d=[np.sum(row) for row in W]
    D=np.diag(d)
    L=D-W
    Dn=np.power(np.linalg.matrix_power(D,-1),0.5)
    Lbar=np.dot(np.dot(Dn,L),Dn)
    return Lbar

def getKSmallestEigVec(Lbar, k):
    """input
    "matrix Lbar and k
    "return
    "k smallest eigen values and their corresponding eigen vectors
    """
    eigval, eigvec = linalg.eig(Lbar)
    dim = len(eigval)

    # 查找前k小的eigval
    dictEigval = dict(zip(eigval, range(0, dim)))
    kEig = np.sort(eigval)[0:k]
    ix = [dictEigval[k] for k in kEig]
    return eigval[ix], eigvec[:,ix]

def accur(p):
    #481 593 585 598 594 546
    label = np.zeros(p.shape[0])
    label[0:481] = 0
    label[481:481+593] = 1
    label[481+593:481+593+585] = 2
    label[481+593+585:481+593+585+598] = 3
    label[481+593+585+598:481+593+585+598+594] = 4
    label[481+593+585+598+594:] = 5
    G = np.random.random((6,6))
    for i in range(6):
        for j in range(6):
            for k in range(p.shape[0]):
                if((p[k]==j)&(label[k]==i)):
                    G[i][j] += 1
    #print(G)
    M = np.zeros((6,6))
    for i in range(6):
        l = np.where(G==np.max(G))
        G[l[0],:] = -1
        G[:,l[1]] = -1
        M[l[0],l[1]] = 1
    #print(M)
    test = np.zeros(p.shape[0])
    test[0:481] = np.argmax(M[:,0])
    test[481:481 + 593] = np.argmax(M[:,1])
    test[481 + 593:481 + 593 + 585] = np.argmax(M[:,2])
    test[481 + 593 + 585:481 + 593 + 585 + 598] = np.argmax(M[:,3])
    test[481 + 593 + 585 + 598:481 + 593 + 585 + 598 + 594] = np.argmax(M[:,4])
    test[481 + 593 + 585 + 598 + 594:] = np.argmax(M[:,5])
    acc = 0
    for i in range(p.shape[0]):
        if(p[i]==test[i]):
            acc+=1
    return acc/p.shape[0]

def accur2(p):
    #584 591 590 578 593
    label = np.zeros(p.shape[0])
    label[0:584] = 0
    label[584:584+591] = 1
    label[584+591:584+591+590] = 2
    label[584+591+590:584+591+590+578] = 3
    label[584+591+590+578:584+591+590+578+593] = 4
    G = np.random.random((5,5))
    for i in range(5):
        for j in range(5):
            for k in range(p.shape[0]):
                if((p[k]==j)&(label[k]==i)):
                    G[i][j] += 1
    #print(G)
    M = np.zeros((5,5))
    for i in range(5):
        l = np.where(G==np.max(G))
        G[l[0],:] = -1
        G[:,l[1]] = -1
        M[l[0],l[1]] = 1
    #print(M)
    test = np.zeros(p.shape[0])

    test[0:584] = np.argmax(M[:, 0])
    test[584:584 + 591] = np.argmax(M[:, 1])
    test[584 + 591:584 + 591 + 590] = np.argmax(M[:, 2])
    test[584 + 591 + 590:584 + 591 + 590 + 578] = np.argmax(M[:, 3])
    test[584 + 591 + 590 + 578:584 + 591 + 590 + 578 + 593] = np.argmax(M[:, 4])
    acc = 0
    for i in range(p.shape[0]):
        if(p[i]==test[i]):
            acc+=1
    return acc/p.shape[0]

D = np.load("SamMatrix_1.npy")

result = np.zeros((50,50))
for k in range(50):
    for r in range(50):
        #K
        data = K(D,66*k)
        #RBF
        data = RBF(data,1/(r*20+1))
        '''
        plt.imshow(data)
        plt.show()
        '''
        #Laplace
        data = getNormLaplacian(data)
        #切分图
        U,qiyi = getKSmallestEigVec(data,5)

        #np.save("qiyi.npy",qiyi)

        #qiyi = np.load("qiyi.npy")

        a = np.zeros([10])
        clf = KMeans(n_clusters=5)
        for i in range(10):
            clf.fit(qiyi)
            label = clf.labels_
            a[i] = accur2(label)
            print("                 "+str(a[i]))
        result[k][r] = np.max(a)
        '''
        f1 = plt.figure(1)
        plt.subplot(211)
        plt.scatter(range(label.shape[0]),label)
        plt.show()
        '''
        print(str(k)+"   "+str(r)+"   "+str(result[k][r]))
    np.save("result2_50.npy",result)