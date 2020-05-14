# T Distribution	

from scipy import special
from scipy.optimize import fsolve
import math


def TDIST_EM(Input_data,v_start=3):
    trans=Input_data
    (N,D)=trans.shape
    #initialize
    u_start=np.mean(trans,axis=0)
    sigma_start=np.cov(trans.transpose())
 
    u_current=u_start
    sigma_current=sigma_start
    v_current=v_start
    for i in range(30):
        #E step
        temp_center_current=trans-u_current
        temp=v_current+np.sum(np.multiply(np.dot(temp_center_current,np.linalg.inv(sigma_current)),temp_center_current),axis=1)
        Exp_h=(v_current+D)/temp#N
        Exp_log_h=special.digamma((v_current+D)/2)-np.log(temp/2)
        
        #M step
        u_next=np.sum(np.multiply(trans.transpose(),Exp_h),axis=1)/np.sum(Exp_h)#D
        temp_center_current=trans-u_next#N*D
        sigma_next=np.dot(np.multiply(temp_center_current.transpose(),Exp_h),temp_center_current)/N
        def f(v):
            return(np.log(v/2)+1-special.digamma(v/2)+np.mean(Exp_log_h-Exp_h))
        v_next=fsolve(f,v_current)
        #check convergence
        delta_u=np.linalg.norm(u_current-u_next)/np.linalg.norm(u_current)
        delta_sigma=np.linalg.norm(sigma_current-sigma_next)/np.linalg.norm(sigma_current)
        delta_v=np.linalg.norm(v_next-v_current)/np.linalg.norm(v_current)




        u_current=u_next
        sigma_current=sigma_next
        v_current=v_next


    u=u_current
    sigma=sigma_current
    v=v_current
    return([u,sigma,v])
    




def Tdist_logP(Input_data_orig,F=True,v_start=3):
    (N,D)=Input_data_orig.shape
    if(F==True):#for F
        Input_data=Input_data_orig
        [u,sigma,v]=TDIST_EM(F_TI)
    else:#for NonF
        Input_data=Input_data_orig
        [u,sigma,v]=TDIST_EM(NonF_TI)
        
    temp_center=Input_data-u
    Tdist_logP_dist=-(1/2)*np.sum(np.log(np.linalg.svd(sigma)[1]))-\
    (v+D)/2*np.log(1+(1/v)*np.sum(np.multiply(np.dot(temp_center,np.linalg.inv(sigma)),temp_center),axis=1))-\
    (D/2)*np.log(math.pi)-(D/2)*np.log(v)-np.log(special.gamma(v/2))+np.log(special.gamma((v+D)/2))
    return(Tdist_logP_dist)
        
    


def Label_T(Input_data_orig,v_start=3,threshold=0.5):
    delta=Tdist_logP(Input_data_orig,F=True,v_start=v_start)-Tdist_logP(Input_data_orig,F=False,v_start=v_start)#log_p_F-log_p_nonF
    ratio_threshold=np.log(threshold/(1-threshold))
    if(isinstance(threshold,np.ndarray)==False):
        estimated_label=np.zeros(Input_data_orig.shape[0])
        estimated_label[[i for i in range(Input_data_orig.shape[0]) if delta[i]>ratio_threshold]]=1
    return(estimated_label)
  


def FR_T(Input_data_orig,tl,v_start=3,threshold=0.5):
    (N,D)=Input_data_orig.shape
    delta=Tdist_logP(Input_data_orig,F=True,v_start=v_start)-Tdist_logP(Input_data_orig,F=False,v_start=v_start)
    ratio_threshold=np.log(threshold/(1-threshold))
    if(isinstance(threshold,np.ndarray)==False):
        # checking whether face or non-face
        estimated_label=np.zeros(N)
        estimated_label[[i for i in range(N) if delta[i]>ratio_threshold]]=1
        #False Rate
        FR=np.zeros(3)
        FR[0]=np.mean(estimated_label[[i for i in range(N) if tl[i]==0]])
        FR[1]=1-np.mean(estimated_label[[i for i in range(N) if tl[i]==1]])
        FR[2]=np.mean(np.abs(estimated_label-tl))
        return(FR)
 


def ROC_T(Input_data_orig,tl,ratio_threshold_seq,v_start=3):  
    N=Input_data_orig.shape[0]
    delta=Tdist_logP(Input_data_orig,F=True,v_start=v_start)-Tdist_logP(Input_data_orig,F=False,v_start=v_start)
    
    if(isinstance(ratio_threshold_seq,np.ndarray)):
        FR=np.zeros((2,len(ratio_threshold_seq)))
        for i in range(len(ratio_threshold_seq)):
            #F_or_nonF
            ratio_threshold=ratio_threshold_seq[i]
            estimated_label=np.zeros(N)
            estimated_label[[i for i in range(N) if delta[i]>ratio_threshold]]=1
            #False Rate
            FR[0,i]=np.mean(estimated_label[[i for i in range(N) if tl[i]==0]])
            FR[1,i]=1-np.mean(estimated_label[[i for i in range(N) if tl[i]==1]])
        plt.plot(FR[0,:],1-FR[1,:],"r--")
        plt.show()             
 
#Evaluate the learned model on the testing images
TETL=np.zeros(200)
TETL[0:100]=1 
TRTL=np.zeros(2000)
TRTL[0:1000]=1 


print(FR_T(TI,tl=TRTL,threshold=0.5,v_start=5))
print(FR_T(TEI,tl=TETL,threshold=0.5,v_start=5))
ROC_T(TEI,tl=TETL,ratio_threshold_seq=np.arange(-1500,1500,100),v_start=5)


#mean
[F_u,F_sigma,F_v]=TDIST_EM(F_TI,v_start=3)
[NonF_u,NonF_sigma,NonF_v]=TDIST_EM(NonF_TI,v_start=3)


plt.subplot(2, 2, 1)
plt.imshow(F_u.reshape((10,10,3)).astype(int))
plt.title("mean-F")


plt.subplot(2, 2, 2)
plt.imshow(NonF_u.reshape((10,10,3)).astype(int))
plt.title("mean-NonFace")


#cov
plt.subplot(2, 2, 3)
cov_diag=np.diag(F_sigma)    
[min_v,max_v]=[np.min(cov_diag),np.max(cov_diag)]
norm_cov_diag=(cov_diag-min_v)/(max_v-min_v)*255
plt.imshow(norm_cov_diag.reshape((10,10,3)).astype(int))
plt.title("cov-Face")


plt.subplot(2, 2, 4)
cov_diag=np.diag(NonF_sigma)  
[min_v,max_v]=[np.min(cov_diag),np.max(cov_diag)]
norm_cov_diag=(cov_diag-min_v)/(max_v-min_v)*255
plt.imshow(norm_cov_diag.reshape((10,10,3)).astype(int))
plt.title("cov-NonFace")

