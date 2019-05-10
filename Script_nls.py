# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 09:26:53 2019

@author: Sam
"""

import scipy as sc;
import numpy as np;
import os
numpy.set_printoptions(threshold=numpy.nan)
import matplotlib as mp
os.system('cls')
mp.pyplot.close('all')
# Input training and test data
train= np.loadtxt("F:\UTA\EE5359_ML\Assignment4\Train.txt", dtype='f', delimiter=',');
train=train[np.argsort(train[:,0])]
train
x=train[:,0];
x_max=max(x)
y=train[:,1];1


Nx=x.size;
Ny=y.size;

Kn=4
Ksplit=Kn+1
ep=(max(x)-min(x))/Kn


Knl1=x[0:Nx/Ksplit]
Knl2=x[(Nx/Ksplit):(Nx/Ksplit)*2]
Knl3=x[(2*Nx/Ksplit):(Nx/Ksplit)*3]
Knl4=x[(3*Nx/Ksplit):(Nx/Ksplit)*4]
Knl5=x[(4*Nx/Ksplit):Nx]

Kn1=Knl1[-1]
Kn2=Knl2[-1]
Kn3=Knl3[-1]
Kn4=Knl4[-1]

d1=(np.maximum(np.power((x-Kn1),3),0)-np.maximum(np.power((x-Kn4),3),0))/(Kn4-Kn1)
d2=(np.maximum(np.power((x-Kn2),3),0)-np.maximum(np.power((x-Kn4),3),0))/(Kn4-Kn2)
d3=(np.maximum(np.power((x-Kn3),3),0)-np.maximum(np.power((x-Kn4),3),0))/(Kn4-Kn3)


N1=np.ones(Nx);
N2=x;
N3=d1-d3;
N4=d2-d3;

N= np.column_stack((N1,N2,N3,N4))



d12= 6*(np.maximum(x-Kn1,0)-np.maximum(x-Kn4,0))/(Kn4-Kn1);
d32= 6*(np.maximum(x-Kn3,0)-np.maximum(x-Kn4,0))/(Kn4-Kn3);
d22= 6*(np.maximum(x-Kn2,0)-np.maximum(x-Kn4,0))/(Kn4-Kn2);

N3_2=d12-d32;
N4_2=d22-d32;

O33=np.sum(N3_2[(Nx/Ksplit):(4*Nx/Ksplit)-1]*N3_2[(Nx/Ksplit):(4*Nx/Ksplit)-1])*(x[(4*Nx/Ksplit)-1]-x[(Nx/Ksplit)-1])/(((4*Nx/Ksplit)-1-(Nx/Ksplit))*Nx)
O34=np.sum(N4_2[(Nx/Ksplit):(4*Nx/Ksplit)-1]*N3_2[(Nx/Ksplit):(4*Nx/Ksplit)-1])*(x[(4*Nx/Ksplit)-1]-x[(Nx/Ksplit)-1])/(((4*Nx/Ksplit)-1-(Nx/Ksplit))*Nx)
O43=np.sum(N3_2[(Nx/Ksplit):(4*Nx/Ksplit)-1]*N4_2[(Nx/Ksplit):(4*Nx/Ksplit)-1])*(x[(4*Nx/Ksplit)-1]-x[(Nx/Ksplit)-1])/(((4*Nx/Ksplit)-1-(Nx/Ksplit))*Nx)
O44=np.sum(N4_2[(Nx/Ksplit):(4*Nx/Ksplit)-1]*N4_2[(Nx/Ksplit):(4*Nx/Ksplit)-1])*(x[(4*Nx/Ksplit)-1]-x[(Nx/Ksplit)-1])/(((4*Nx/Ksplit)-1-(Nx/Ksplit))*Nx)

O=np.array([[0,0,0,0],[0,0,0,0],[0,0,O33,O34],[0,0,O43,O44]])

lamda=0;
#Q=np.matmul(np.matmul((np.linalg.inv((np.matmul(np.transpose(N),N))+(lamda*O))),(np.transpose(N))),y)





################################### Loop to calculate EPE for different values of lambda########################
Err_vec=np.zeros(25)
Err_vec=[]
c=0
for kk in range (0,50,2):
    Err_est=0;
    c=c+1
    #print kk
    for i in range(0,Nx):
        
        
        y_cv=np.delete(y,i);
        x_cv=np.delete(x,i);
        Q=NCS(x_cv, y_cv, kk);
        d1=(np.maximum(np.power((x[i]-Kn1),3),0)-np.maximum(np.power((x[i]-Kn4),3),0))/(Kn4-Kn1)
        d2=(np.maximum(np.power((x[i]-Kn2),3),0)-np.maximum(np.power((x[i]-Kn4),3),0))/(Kn4-Kn2)
        d3=(np.maximum(np.power((x[i]-Kn3),3),0)-np.maximum(np.power((x[i]-Kn4),3),0))/(Kn4-Kn3)
    
    
        N1=1;
        N2=x[i];
        N3=d1-d3;
        N4=d2-d3;
        #print N3
        Nm= [N1, N2, N3, N4]
        Err_est=Err_est+((y[i]-np.matmul(Q,Nm))**2)
        #print y[i], np.matmul(Q,Nm)
        #print "\n"
        
    Err_vec=np.hstack((Err_vec, Err_est))
mp.pyplot.figure(100);
mp.pyplot.xlabel('lambda');
mp.pyplot.ylabel('EPE')
mp.pyplot.grid();
mp.pyplot.title("Lambda v/s EPE curve")
mp.pyplot.plot(range(0,50,2),Err_vec/Nx, '-g')

print "EPE values are" + str(Err_vec/Nx)
Err_vec_min=min(Err_vec/Nx)
print "Minimum EPE occured is "+ str(Err_vec_min)

Err_arg_min=np.argmin(Err_vec)
lamda=(range(0,50,2))[Err_arg_min]
print "Value of lambda where minimum value for EPE occur is "+ str(lamda);


cross=(np.linalg.inv((np.matmul(np.transpose(N),N))+(lamda*(O))))
prdt=np.matmul(np.transpose(N), y);
Q=np.matmul(cross, prdt);

Deg1=np.matmul(cross,np.transpose(N))
Deg2=np.matmul(N, Deg1)
print ("Degrees of Freedom is = " +str(np.trace(Deg2)))
f_est=np.matmul(N,Q)
mp.pyplot.figure(1);
mp.pyplot.plot(x, f_est, '-r', label="Natural Cubic Spline fit for lamda = "+ str(lamda))
mp.pyplot.xlabel('x');
mp.pyplot.ylabel('y')
mp.pyplot.scatter(x, y)
mp.pyplot.legend();
mp.pyplot.grid();




###############Function for Natural cubic Spline##################################33
def NCS(x,y, lamda):
    Nx=x.size;
    #Ny=y.size;
    
    Kn=4
    Ksplit=Kn+1
    #ep=(max(x)-min(x))/Kn
    
    
    Knl1=x[0:Nx/Ksplit]
    Knl2=x[(Nx/Ksplit):(Nx/Ksplit)*2]
    Knl3=x[(2*Nx/Ksplit):(Nx/Ksplit)*3]
    Knl4=x[(3*Nx/Ksplit):(Nx/Ksplit)*4]
    #Knl5=x[(4*Nx/Ksplit):Nx]
    
    Kn1=Knl1[-1]
    Kn2=Knl2[-1]
    Kn3=Knl3[-1]
    Kn4=Knl4[-1]
    
    d1=(np.maximum(np.power((x-Kn1),3),0)-np.maximum(np.power((x-Kn4),3),0))/(Kn4-Kn1)
    d2=(np.maximum(np.power((x-Kn2),3),0)-np.maximum(np.power((x-Kn4),3),0))/(Kn4-Kn2)
    d3=(np.maximum(np.power((x-Kn3),3),0)-np.maximum(np.power((x-Kn4),3),0))/(Kn4-Kn3)
    
    
    N1=np.ones(Nx);
    N2=x;
    N3=d1-d3;
    N4=d2-d3;
    
    N= np.column_stack((N1,N2,N3,N4))
    
    
    
    d12= 6*(np.maximum(x-Kn1,0)-np.maximum(x-Kn4,0))/(Kn4-Kn1);
    d32= 6*(np.maximum(x-Kn3,0)-np.maximum(x-Kn4,0))/(Kn4-Kn3);
    d22= 6*(np.maximum(x-Kn2,0)-np.maximum(x-Kn4,0))/(Kn4-Kn2);
    
    N3_2=d12-d32;
    N4_2=d22-d32;
    
    O33=np.sum(N3_2[(Nx/Ksplit):(4*Nx/Ksplit)-1]*N3_2[(Nx/Ksplit):(4*Nx/Ksplit)-1])*(x[(4*Nx/Ksplit)-1]-x[(Nx/Ksplit)-1])/(((4*Nx/Ksplit)-1-(Nx/Ksplit))*Nx)
    O34=np.sum(N4_2[(Nx/Ksplit):(4*Nx/Ksplit)-1]*N3_2[(Nx/Ksplit):(4*Nx/Ksplit)-1])*(x[(4*Nx/Ksplit)-1]-x[(Nx/Ksplit)-1])/(((4*Nx/Ksplit)-1-(Nx/Ksplit))*Nx)
    O43=np.sum(N3_2[(Nx/Ksplit):(4*Nx/Ksplit)-1]*N4_2[(Nx/Ksplit):(4*Nx/Ksplit)-1])*(x[(4*Nx/Ksplit)-1]-x[(Nx/Ksplit)-1])/(((4*Nx/Ksplit)-1-(Nx/Ksplit))*Nx)
    O44=np.sum(N4_2[(Nx/Ksplit):(4*Nx/Ksplit)-1]*N4_2[(Nx/Ksplit):(4*Nx/Ksplit)-1])*(x[(4*Nx/Ksplit)-1]-x[(Nx/Ksplit)-1])/(((4*Nx/Ksplit)-1-(Nx/Ksplit))*Nx)
    
    O=np.array([[0,0,0,0],[0,0,0,0],[0,0,O33,O34],[0,0,O43,O44]])
    
    #lamda=0;
    #Q=np.matmul(np.matmul((np.linalg.inv((np.matmul(np.transpose(N),N))+(lamda*O))),(np.transpose(N))),y)
    
    cross=(np.linalg.inv((np.matmul(np.transpose(N),N))+(lamda*(O))))
    prdt=np.matmul(np.transpose(N), y);
    Q=np.matmul(cross, prdt);
    f_est=np.matmul(N,Q)
    
    
    #print Q
    return (Q)
    
    
    
   
        




