# Factor Analysis	

import matplotlib.pyplot as plt


def Factor_EM(Input_data,K_sub=3):
    trans=Input_data
    (N,D)=trans.shape
    #Initialize
    u=np.mean(trans,axis=0)
    Input_data=FTI
    sigma_full=np.cov(FTI.transpose())
    [U_matrix,D_matrix,V_matrix]=np.linalg.svd(sigma_full)
    eta_start=np.multiply(np.sqrt(D_matrix[0:K_sub]),U_matrix[:,0:K_sub])#
    sigma_start=np.diag(np.diag(sigma_full)-np.diag(np.dot(eta_start,eta_start.transpose())))
    
    sigma_current=sigma_start
    eta_current=eta_start
    
    Exp_hh=np.zeros((N,K_sub,K_sub))
    sum_Exp_hh=np.zeros((K_sub,K_sub))
    
    for t in range(60):
        
        sigma_current_inv=np.linalg.pinv(sigma_current)
        a=np.dot(np.dot(eta_current.transpose(),sigma_current_inv),eta_current)
        
        b=np.linalg.pinv(a+np.identity(K_sub))
        d=trans-u
        
        Exp_h=np.dot(np.dot(np.dot(b,eta_current.transpose()),sigma_current_inv),d.transpose()).transpose()#N*K
 
        for i in range(N):
            Exp_hh[i,:,:]= np.outer(Exp_h[i,:],Exp_h[i,:])+b
            sum_Exp_hh=sum_Exp_hh+Exp_hh[i,:,:]
        
        e=np.dot(Exp_h.transpose(),d)
       
        eta_next=np.dot(e.transpose(),np.linalg.pinv(sum_Exp_hh))
        sigma_next=np.diag(np.diag(np.dot(d.transpose(),d))-np.diag(np.dot(eta_next,e)))/N#D*D
        
        #checking convergence
        delta_eta=np.linalg.norm(eta_next-eta_current)/np.linalg.norm(eta_current)
        delta_sigma=np.linalg.norm(sigma_next-sigma_current)/np.linalg.norm(sigma_current)



        eta_current=eta_next
        sigma_current=sigma_next
        
    eta=eta_current
    sigma=sigma_current
    
    return([u,eta,sigma])


  
def Factor_logP(Input_data_orig,F=True,K_sub=3):
    if(F==True):#for F
        Input_data=Input_data_orig
        [u,eta,sigma]=Factor_EM(FTI,K_sub=K_sub)
    else:#for NonF
        Input_data=Input_data_orig
        [u,eta,sigma]=Factor_EM(NonFTI,K_sub=K_sub)
        
    temp_1=np.dot(eta,eta.transpose())+sigma
    temp_2=Input_data-u
    log_p=-(1/2)*np.sum(np.log(np.linalg.svd(temp_1)[1]))-(1/2)*np.sum(np.multiply(np.dot(temp_2,np.linalg.pinv(temp_1)),temp_2),axis=1)
    return(log_p)
    
def Label_Factor(Input_data_orig,K_sub=3,threshold=0.5):
    delta=Factor_logP(Input_data_orig,F=True,K_sub=K_sub)-Factor_logP(Input_data_orig,F=False,K_sub=K_sub)
    rt=np.log(threshold/(1-threshold))
    
    if(isinstance(threshold,np.ndarray)==False):
        estimated_label=np.zeros(Input_data_orig.shape[0])
        estimated_label[[i for i in range(Input_data_orig.shape[0]) if delta[i]>rt]]=1
    return(estimated_label)
    
#FR
def FR_Factor(Input_data_orig,true_label,K_sub=3,threshold=0.5):
    N=Input_data_orig.shape[0]
    delta=Factor_logP(Input_data_orig,F=True,K_sub=K_sub)-Factor_logP(Input_data_orig,F=False,K_sub=K_sub)#log_p_F-log_p_nonF
    rt=np.log(threshold/(1-threshold))
    if(isinstance(threshold,np.ndarray)==False):#threshold is a scalar
        #F_or_nonF
        estimated_label=np.zeros(N)
        estimated_label[[i for i in range(N) if delta[i]>rt]]=1
        #False Rate
        FR=np.zeros(3)
        FR[0]=np.mean(estimated_label[[i for i in range(N) if true_label[i]==0]])
        FR[1]=1-np.mean(estimated_label[[i for i in range(N) if true_label[i]==1]])
        FR[2]=np.mean(np.abs(estimated_label-true_label))
        return(FR)



def Factor_ROC(Input_data_orig,true_label,rt_seq,K_sub=3):
    N=Input_data_orig.shape[0]
    delta=Factor_logP(Input_data_orig,F=True,K_sub=K_sub)-Factor_logP(Input_data_orig,F=False,K_sub=K_sub)#log_p_F-log_p_nonF
    if(isinstance(rt_seq,np.ndarray)):#threshold is a seq
        FR=np.zeros((2,len(rt_seq)))
        for i in range(len(rt_seq)):
            #F_or_nonF
            rt=rt_seq[i]
            estimated_label=np.zeros(N)
            estimated_label[[i for i in range(N) if delta[i]>rt]]=1
            #False Rate
            FR[0,i]=np.mean(estimated_label[[i for i in range(N) if true_label[i]==0]])
            FR[1,i]=1-np.mean(estimated_label[[i for i in range(N) if true_label[i]==1]])
        plt.plot(FR[0,:],1-FR[1,:],"r--")
        plt.show()
        
#Evaluating the learned model on the testing images
Test_true_label=np.zeros(200)
Test_true_label[0:100]=1   
TTL=np.zeros(2000)
TTL[0:1000]=1 




print(FR_Factor(TI,true_label=TTL,threshold=0.5,K_sub=3))
print(FR_Factor(Test_images,true_label=Test_true_label,threshold=0.5,K_sub=3))
Factor_ROC(TI,true_label=TTL,rt_seq=np.arange(-1500,1500,100),K_sub=3)




[F_u,F_eta,F_sigma]=Factor_EM(FTI,3)
[NonF_u,NonF_eta,NonF_sigma]=Factor_EM(NonFTI,3)


#mean
plt.subplot(2, 2, 1)
plt.imshow(F_u.reshape((10,10,3)).astype(int))
plt.title("mean-Face")

plt.subplot(2, 2, 2)
plt.imshow(NonF_u.reshape((10,10,3)).astype(int))
plt.title("mean-NonFace")


#covariance
plt.subplot(2, 2, 3)
cov_diag=np.diag(np.dot(F_eta,F_eta.transpose())+F_sigma)    
[min_v,max_v]=[np.min(cov_diag),np.max(cov_diag)]
norm_cov_diag=(cov_diag-min_v)/(max_v-min_v)*255
plt.imshow(norm_cov_diag.reshape((10,10,3)).astype(int))
plt.title("cov-Face")

plt.subplot(2, 2, 4)
cov_diag=np.diag(np.dot(NonF_eta,NonF_eta.transpose())+NonF_sigma) 
[min_v,max_v]=[np.min(cov_diag),np.max(cov_diag)]
norm_cov_diag=(cov_diag-min_v)/(max_v-min_v)*255
plt.imshow(norm_cov_diag.reshape((10,10,3)).astype(int))
plt.title("cov-NonFace")


# FLASE POSITIVE RATE, FALSE NEGATIVE RATE AND MISCLASSIFICATION RATE FOR ALL THE MODELS




print(FR_Gaussian(Test_images,true_label=Test_true_label,threshold=0.5))


print(FR_Mix_Gaussian(Test_images,Test_true_label,K=6,threshold=0.5))


print(FR_T(Test_images,true_label=Test_true_label,threshold=0.5,v_start=5))


print(FR_Factor(Test_images,true_label=Test_true_label,threshold=0.5,K_sub=3))

