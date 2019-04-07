import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def costError(X,Y,theta):
    j=(1/(2*row))*(np.sum(np.power(X*theta.T-Y,2))) 
    return j
#########################################################
def gradientDescent(X,Y,theta,alpha,m,ite):
    hx=theta*X.T
    j=np.zeros(ite)
    for i in range(ite):
        hx=X*theta.T
        temp1=theta[0,0]-alpha*(1/m)*np.sum(hx-Y)
        temp2=theta[0,1]-alpha*(1/m)*((((hx-Y).T)*X[:,1]))
        theta[0,0]=temp1
        theta[0,1]=temp2
        j[i]=costError(X,Y,theta)
    return theta,j
################################################
path="ex1data1.txt"
datarp=pd.read_csv(path,header=None,names=['input','output'])
#datarp=(datarp-datarp.mean())/datarp.std()
#datarp.insert(0,"ones",1)
#plt.scatter(datarp['input'],datarp['output'])
#plt.xlabel("population")
#plt.ylabel("profit")
#X=np.matrix(datarp.iloc[:,:2])
#Y=np.matrix(datarp.iloc[:,2:3])
#row=np.size(X,axis=0)
#alpha=0.1
#theta=np.zeros((1,2))
#theta,j=gradientDescent(X,Y,theta,alpha,row,1000)
#f=X*theta.T
#plt.plot(X[:,1],f,"r")
#fig,ax=plt.subplots(figsize=(5,5))
#ax.set_title("represtnation of error")
#ax.plot(np.arange(1000),j,"r")

