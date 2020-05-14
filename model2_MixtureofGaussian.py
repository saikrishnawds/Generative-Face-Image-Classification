
#Mixture of Gaussian



def MOGM_EM(Input_data,K=3): 
    trans=Input_data
    (N,D)=trans.shape
    
    h_start=np.ones(K)*(1/K)
    u_start=np.ones((K,D))
    sigma_start=np.ones((K,D,D))
    a=list(range(N))
    random.shuffle(a)
    group_size=int(N/K)
    for i in range(K):
        u_start[i,:]=np.mean(trans[a[(group_size*i):(group_size*(i+1))],:],axis=0)
        sigma_start[i,:,:]=np.diag(np.diag(np.cov(trans[a[(group_size*i):(group_size*(i+1))],].transpose())))#D*D
    h_next=np.zeros(K)
    u_next=np.zeros((K,D))
    sigma_next=np.zeros((K,D,D))
    h_current=h_start
    u_current=u_start
    sigma_current=sigma_start
    
    Gauss_logx=np.ones((N,K))
    S=np.ones((N,K))
    #EM starts here
    for t in range(30):
        #E-step below 
        for k in range(K):
            temp_center=trans-u_current[k,:]
            Gauss_logx[:,k]=-(1/2)*np.sum(np.multiply(np.dot(temp_center,np.linalg.pinv(sigma_current[k,:,:])),temp_center),axis=1)-\
            (1/2)*np.sum(np.log(np.linalg.svd(sigma_current[k,:,:])[1]))
        for k in range(K):
            for n in range(N):
                S[n,k]=h_current[k]/np.sum(h_current*np.exp(Gauss_logx[n,:]-Gauss_logx[n,k]))  
        #M-step and updating u
        
        u_next=(np.dot(S.transpose(),trans).transpose()/np.sum(S,axis=0)).transpose()#K*D
        #updating sigma
        for k in range(K):
           temp_center=trans-u_next[k,:]
           sigma_next[k,:,:]=np.dot(np.multiply(temp_center.transpose(),S[:,k]),temp_center)/np.sum(S[:,k])
           sigma_next[k,:,:]-sigma_current[k,:,:]
        #updating h  
        h_next=np.sum(S,axis=0)/np.sum(S)
        
        #checking convergence
        delta_u=np.linalg.norm(u_next-u_current)/np.linalg.norm(u_current)
        delta_h=np.linalg.norm(h_next-h_current)/np.linalg.norm(h_current)
        delta_sigma=np.linalg.norm(sigma_next-sigma_current)/np.linalg.norm(sigma_current)
        #print(np.linalg.norm(u_next-u_current))
        #print(delta_u)
        #print(delta_h)
        #print(delta_sigma)
      
        u_current=u_next
        h_current=h_next
        for k in range(K):
           sigma_current[k,:,:]=np.diag(np.diag(sigma_next[k,:,:]))
        
    sigma=sigma_current
    h=h_current
    u=u_current
    return([sigma,h,u])




def MOGM_label(Input_data_orig,K,threshold=0.5):
    N=Input_data_orig.shape[0]
    (F_sigma,F_h,F_u)=MOGM_EM(F_Train_images,K=K)   
    (NonF_sigma,NonF_h,NonF_u)=MOGM_EM(NonF_Train_images,K=K) 
    F_trans=Input_data_orig#N*D
    NonF_trans=Input_data_orig
    
 
    temp_F=np.zeros((Input_data_orig.shape[0],K))
    temp_nonF=np.zeros((Input_data_orig.shape[0],K))
    for i in range(K):
        temp_F[:,i]=np.sum(np.multiply(np.dot(F_trans-F_u[i,:],np.linalg.pinv(F_sigma[i,:,:])),F_trans-F_u[i,:]),axis=1)
        temp_nonF[:,i]=np.sum(np.multiply(np.dot(NonF_trans-NonF_u[i,:],np.linalg.pinv(NonF_sigma[i,:,:])),NonF_trans-NonF_u[i,:]),axis=1)   
    log_det_F=np.zeros(K)
    log_det_nonF=np.zeros(K)
    for i in range(K):
        log_det_F[i]=np.sum(np.log(np.linalg.svd(F_sigma[i,:,:])[1]))
        log_det_nonF[i]=np.sum(np.log(np.linalg.svd(NonF_sigma[i,:,:])[1])) 
     
    estimated_label=np.ones(Input_data_orig.shape[0])*2
    
    
    if(False):
        p_ratio_F_non=np.zeros(Input_data_orig.shape[0])
        for n in range(Input_data_orig.shape[0]):
            #n=0
            temp_p=np.zeros(K)
            for j in range(K):    
                temp_p[j]=np.sum(np.exp(-(1/2)*(log_det_nonF-log_det_F[j])-(1/2)*(temp_nonF[n,:]-temp_F[n,j])+np.log(NonF_h)-np.log(F_h[j])))
            #print(temp_p)
            p_ratio_F_non[n]=np.sum(1.0/temp_p)


        
        p_ratio_non_F=np.zeros(Input_data_orig.shape[0])
        for n in range(Input_data_orig.shape[0]):
            #n=0
            temp_p=np.zeros(K)
            for j in range(K):
                temp_p[j]=np.sum(np.exp(-(1/2)*(log_det_F-log_det_nonF[j])-(1/2)*(temp_F[n,:]-temp_nonF[n,j])*(F_h/NonF_h[j])))
            p_ratio_non_F[n]=np.sum(1.0/temp_p)
           
        p_ratio_non_F
        p_ratio_F_non
        useful_index=[i for i in range(Input_data_orig.shape[0]) if p_ratio_non_F[i]!=p_ratio_F_non[i]]
        no_useful_index=[i for i in range(Input_data_orig.shape[0]) if p_ratio_non_F[i]==p_ratio_F_non[i]]
        estimated_label[[i for i in useful_index if p_ratio_non_F[i]!=p_ratio_F_non[i] and p_ratio_F_non[i]>((1-threshold)/threshold)]]=1
        estimated_label[[i for i in useful_index if p_ratio_non_F[i]!=p_ratio_F_non[i] and p_ratio_F_non[i]<=((1-threshold)/threshold)]]=0
        
    
    log_p_ratio_F_non=np.zeros(Input_data_orig.shape[0])
    for n in range(Input_data_orig.shape[0]):
        temp_p=np.zeros(K)
        for j in range(K): 
            temp_p[j]=np.max(-(1/2)*(log_det_nonF-log_det_F[j])-(1/2)*(temp_nonF[n,:]-\
                  temp_F[n,j])+np.log(NonF_h)-np.log(F_h[j]))
        log_p_ratio_F_non[n]=-np.min(temp_p)
    log_p_ratio_non_F=np.zeros(Input_data_orig.shape[0])
    for n in range(Input_data_orig.shape[0]):
        temp_p=np.zeros(K)
        for j in range(K):
            temp_p[j]=np.max(-(1/2)*(log_det_F-log_det_nonF[j])-(1/2)*(temp_F[n,:]-\
                  temp_nonF[n,j])+np.log(F_h)-np.log(NonF_h[j]))
        log_p_ratio_non_F[n]=-np.min(temp_p)


    
    estimated_label[[i for i in range(N) if log_p_ratio_F_non[i]>-np.log(threshold/(1-threshold))]]=1
    estimated_label[[i for i in range(N) if log_p_ratio_F_non[i]<=-np.log(threshold/(1-threshold))]]=0


    return(estimated_label)
    
  
    
def FR_Mix_Gaussian(Input_data_orig,true_label,K=3,threshold=0.5):
    N=Input_data_orig.shape[0]
    estimated_label=MOGM_label(Input_data_orig,K=K,threshold=threshold)
    ratio_threshold=np.log(threshold/(1-threshold))
    if(isinstance(threshold,np.ndarray)==False):      
        FR=np.zeros(3)
        FR[0]=np.mean(estimated_label[[i for i in range(N) if true_label[i]==0]])
        FR[1]=1-np.mean(estimated_label[[i for i in range(N) if true_label[i]==1]])
        FR[2]=np.mean(np.abs(estimated_label-true_label))
        return(FR)
    


    
def ROC_Mix_Gaussian(Input_data_orig,true_label,ratio_threshold_seq,K=3):
    N=Input_data_orig.shape[0]
    (F_sigma,F_h,F_u)=MOGM_EM(F_Train_images,K=K)   
    (NonF_sigma,NonF_h,NonF_u)=MOGM_EM(NonF_Train_images,K=K) 
    F_trans=Input_data_orig
    NonF_trans=Input_data_orig  
    #(x-u)Sigma^(-1)(x-u)
    temp_F=np.zeros((Input_data_orig.shape[0],K))#N*K
    temp_nonF=np.zeros((Input_data_orig.shape[0],K))
    for i in range(K):
        temp_F[:,i]=np.sum(np.multiply(np.dot(F_trans-F_u[i,:],np.linalg.pinv(F_sigma[i,:,:])),F_trans-F_u[i,:]),axis=1)
        temp_nonF[:,i]=np.sum(np.multiply(np.dot(NonF_trans-NonF_u[i,:],np.linalg.pinv(NonF_sigma[i,:,:])),NonF_trans-NonF_u[i,:]),axis=1)   


    log_det_F=np.zeros(K)
    log_det_nonF=np.zeros(K)
    for i in range(K):
        log_det_F[i]=np.sum(np.log(np.linalg.svd(F_sigma[i,:,:])[1]))
        log_det_nonF[i]=np.sum(np.log(np.linalg.svd(NonF_sigma[i,:,:])[1])) 
    if(False):
        #no numerical problems
        p_ratio_F_non=np.zeros(Input_data_orig.shape[0])
        for n in range(Input_data_orig.shape[0]):
            temp_p=np.zeros(K)
            for j in range(K):    
                temp_p[j]=np.sum(np.exp(-(1/2)*(log_det_nonF-log_det_F[j])-(1/2)*(temp_nonF[n,:]-temp_F[n,j])+np.log(NonF_h)-np.log(F_h[j])))


            p_ratio_F_non[n]=1/np.sum(temp_p)        
        p_ratio_non_F=np.zeros(Input_data_orig.shape[0])
        for n in range(Input_data_orig.shape[0]):
            #n=0
            temp_p=np.zeros(K)
            for j in range(K):
                temp_p[j]=np.sum(np.exp(-(1/2)*(log_det_F-log_det_nonF[j])-(1/2)*(temp_F[n,:]-temp_nonF[n,j])*(F_h/NonF_h[j])))
            p_ratio_non_F[n]=1/np.sum(temp_p)
          
     #numerical problem exists
    log_p_ratio_F_non=np.zeros(Input_data_orig.shape[0])
    for n in range(Input_data_orig.shape[0]):
        temp_p=np.zeros(K)
        for j in range(K): 
            #j=0
            temp_p[j]=np.max(-(1/2)*(log_det_nonF-log_det_F[j])-(1/2)*(temp_nonF[n,:]-\
                  temp_F[n,j])+np.log(NonF_h)-np.log(F_h[j]))
            #print(temp_p[j])
            
        #print(temp_p)
        
        log_p_ratio_F_non[n]=-np.min(temp_p)
        #print(log_p_ratio_F_non[n])
    #print(log_p_ratio_F_non)
    
    log_p_ratio_non_F=np.zeros(Input_data_orig.shape[0])
    for n in range(Input_data_orig.shape[0]):
        #n=0
        temp_p=np.zeros(K)
        for j in range(K):
            temp_p[j]=np.max(-(1/2)*(log_det_F-log_det_nonF[j])-(1/2)*(temp_F[n,:]-\
                  temp_nonF[n,j])+np.log(F_h)-np.log(NonF_h[j]))
        #print(temp_p)
        log_p_ratio_non_F[n]=-np.min(temp_p)
        #print(log_p_ratio_non_F[n])
    #print(log_p_ratio_non_F)
    FR=np.zeros((2,len(ratio_threshold_seq)))
    for i in range(len(ratio_threshold_seq)):
        #F_or_nonF
        ratio_threshold=ratio_threshold_seq[i]
        estimated_label=np.ones(Input_data_orig.shape[0])*2
        estimated_label[[i for i in range(Input_data_orig.shape[0]) if log_p_ratio_F_non[i]>-ratio_threshold]]=1#>-np.log(threshold/(1-threshold))
        estimated_label[[i for i in range(Input_data_orig.shape[0]) if log_p_ratio_F_non[i]<=-ratio_threshold]]=0#<=-np.log(threshold/(1-threshold)
       
        FR[0,i]=np.mean(estimated_label[[i for i in range(N) if true_label[i]==0]])
        FR[1,i]=1-np.mean(estimated_label[[i for i in range(N) if true_label[i]==1]])
    plt.plot(FR[0,:],1-FR[1,:],"r--")
    plt.show()
        


#Evaluating the learned model on the testing images
Test_true_label=np.zeros(200)
Test_true_label[0:100]=1    
Train_true_label=np.zeros(2000)
Train_true_label[0:1000]=1


for K in range(1,7):
    print(K)
    print(FR_Mix_Gaussian(Train_images,Train_true_label,K=K,threshold=0.5))
    
for K in range(1,7):
    print(K)
    print(FR_Mix_Gaussian(Test_images,Test_true_label,K=K,threshold=0.5))
  


(F_sigma,F_h,F_u)=MOGM_EM(F_Train_images,K=5)   
(NonF_sigma,NonF_h,NonF_u)=MOGM_EM(NonF_Train_images,K=5) 




ROC_Mix_Gaussian(Test_images,true_label=Test_true_label,ratio_threshold_seq=np.arange(-1500,1500,100),K=5)
ROC_Mix_Gaussian(Test_images,true_label=Test_true_label,ratio_threshold_seq=np.arange(-1500,1500,100),K=5)




#mean
(F_sigma,F_h,F_u)=MOGM_EM(F_Train_images,K=K)   
(NonF_sigma,NonF_h,NonF_u)=MOGM_EM(NonF_Train_images,K=K) 


plt.imshow(np.dot(F_h,F_u).reshape((10,10,3)).astype(int))
plt.title("mean-Face")


plt.imshow(np.dot(NonF_h,NonF_u).reshape((10,10,3)).astype(int))
plt.title("mean-NonFace")




#cov
cov_diag=np.zeros(10*10*3)
for i in range(F_sigma.shape[0]):
    cov_diag=cov_diag+np.diag(F_sigma[i,:,:])*F_h[i]
    
[min_v,max_v]=[np.min(cov_diag),np.max(cov_diag)]
norm_cov_diag=(cov_diag-min_v)/(max_v-min_v)*255
plt.imshow(norm_cov_diag.reshape((10,10,3)).astype(int))
plt.title("cov-F")


cov_diag=np.zeros(10*10*3)
for i in range(F_sigma.shape[0]):
    cov_diag=cov_diag+np.diag(NonF_sigma[i,:,:])*NonF_h[i]
[min_v,max_v]=[np.min(cov_diag),np.max(cov_diag)]
norm_cov_diag=(cov_diag-min_v)/(max_v-min_v)*255
plt.imshow(norm_cov_diag.reshape((10,10,3)).astype(int))
plt.title("cov-NonFace")

