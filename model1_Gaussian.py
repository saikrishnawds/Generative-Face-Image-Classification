# Single Gaussian	

import os
import matplotlib.image as mpimg 
import numpy as np
import matplotlib.pyplot as plt
import random
import numpy as np




def GaussEM(Input_data):
    trans=Input_data
    mu=np.mean(trans,axis=0)
    sigma=np.cov(trans.transpose())
    return([mu,sigma])




def Log_p_Gauss(Input_data_orig,Face=True):
    if(Face==True):#for Face
        [mu,sigma]=GaussEM(Face_Train_images)
    else:#for Nonface
        [mu,sigma]=GaussEM(Nonface_Train_images)
    temp_center=Input_data_orig-mu
    log_p=np.sum(np.log(np.linalg.svd(sigma)[1]))*(-1/2)-(1/2)*np.sum(np.multiply(np.dot(temp_center,np.linalg.pinv(sigma)),temp_center),axis=1)
    return(log_p)




def Label_Gaussian(Input_data,threshold=0.5):
    delta=Log_p_Gauss(Input_data,Face=True)-Log_p_Gauss(Input_data,Face=False)#log_p_face-log_p_nonface
    rt=np.log(threshold/(1-threshold))
    if(isinstance(threshold,np.ndarray)==False):
        estimated_label=np.zeros(Input_data.shape[0])
        estimated_label[[i for i in range(Input_data.shape[0]) if delta[i]>rt]]=1
    return(estimated_label)
    


def Gauss_FR(Input_data,true_label,threshold=0.5):
    N=Input_data.shape[0]
    delta=Log_p_Gauss(Input_data,Face=True)-Log_p_Gauss(Input_data,Face=False)#log_p_face-log_p_nonface
    rt=np.log(threshold/(1-threshold))
    if(isinstance(threshold,np.ndarray)==False):#threshold is a scalar
        #face_or_nonface
        estimated_label=np.zeros(N)
        estimated_label[[i for i in range(N) if delta[i]>rt]]=1
        #False Rate
        FR=np.zeros(3)
        FR[0]=np.mean(estimated_label[[i for i in range(N) if true_label[i]==0]])
        FR[1]=1-np.mean(estimated_label[[i for i in range(N) if true_label[i]==1]])
        FR[2]=np.mean(np.abs(estimated_label-true_label))
        return(FR)
                


def Gauss_ROC(Input_data,true_label,rts):
    N=Input_data.shape[0]
    delta=Log_p_Gauss(Input_data,Face=True)-Log_p_Gauss(Input_data,Face=False)
    if(isinstance(rts,np.ndarray)):
        FR=np.zeros((2,len(rts)))
        for i in range(len(rts)):
            #face_or_nonface
            rt=rts[i]
            estimated_label=np.zeros(N)
            estimated_label[[i for i in range(N) if delta[i]>rt]]=1
            #False Rate
            FR[0,i]=np.mean(estimated_label[[i for i in range(N) if true_label[i]==0]])
            FR[1,i]=1-np.mean(estimated_label[[i for i in range(N) if true_label[i]==1]])
        plt.plot(FR[0,:],1-FR[1,:],"r--")
        plt.xlabel("false positive rate")
        plt.ylabel("true positive rate")
        plt.title("ROC-Gaussian")
        plt.show()
 
#Evaluating  the learned model on the testing images
Test_true_label=np.zeros(200)
Test_true_label[0:100]=1 
Train_true_label=np.zeros(2000)  
Train_true_label[0:1000]=1 


#GaussEM(Face_Train_trans)
print(Gauss_FR(Train_images,true_label=Train_true_label,threshold=0.5))
print(Gauss_FR(Test_images,true_label=Test_true_label,threshold=0.5))
Gauss_ROC(Test_images,true_label=Test_true_label,rts=np.arange(-1500,1500,100))


#mean
[Face_mu,Face_sigma]=GaussEM(Face_Train_images)
plt.imshow(Face_mu.reshape((10,10,3)).astype(int))
plt.title("mean-Face")


[Nonface_mu,Nonface_sigma]=GaussEM(Nonface_Train_images)
plt.imshow(Nonface_mu.reshape((10,10,3)).astype(int))
plt.title("mean-NonFace")


#covariance
cov_diag=np.diag(Face_sigma)
[min_v,max_v]=[np.min(cov_diag),np.max(cov_diag)]
norm_cov_diag=(cov_diag-min_v)/(max_v-min_v)*255
plt.imshow(norm_cov_diag.reshape((10,10,3)).astype(int))
plt.title("cov-Face")


cov_diag=np.diag(Nonface_sigma)
[min_v,max_v]=[np.min(cov_diag),np.max(cov_diag)]
norm_cov_diag=(cov_diag-min_v)/(max_v-min_v)*255
plt.imshow(norm_cov_diag.reshape((10,10,3)).astype(int))
plt.title("cov-NonFace")




        

