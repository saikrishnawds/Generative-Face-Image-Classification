	
import os
#from scipy import misc
import matplotlib.image as mpimg 
import numpy as np
import matplotlib.pyplot as plt
import random



# Reading Train Face images and Nonface images into array


os.getcwd()
#os.chdir("Documents/ncsu course/ncsu 2019 spring/ECE/Project 1")
res=10

TRI=os.listdir("res"+str(res)+"by"+str(res)+"/extracted_pics/Train")
print(len(TRI))
print(len(os.listdir("res"+str(res)+"by"+str(res)+"/extracted_pics/Train")))#2000 images


FTRI=[name for name in TRI if name.startswith("face_")==True]
NonFTRI=[name for name in TRI if name.startswith("nonface_")==True]


print(len(FTRI))
print(len(NonFTRI))


FTRI_arr=np.zeros((1000,res,res,3))
for i in range(1000):# 3 doesn'r work
    #print(i)
    #type(i)
    rbg_arr=mpimg.imread("res"+str(res)+"by"+str(res)+"/extracted_pics/Train/"+FTRI[i])
    
    if rbg_arr.shape!=(res,res,3):
        print("False")
    FTRI_arr[i,:,:,:]=rbg_arr
    
    
NonFTRI_arr=np.zeros((1000,res,res,3))
for i in range(1000):# 3 doesn'r work
    #print(i)
    #print(NonFTRI[i])
    #type(i)
    rbg_arr=mpimg.imread("res"+str(res)+"by"+str(res)+"/extracted_pics/Train/"+NonFTRI[i])
    
    if rbg_arr.shape!=(res,res,3):
        print("False")
    NonFTRI_arr[i,:,:,:]=rbg_arr
   

FTRI=FTRI_arr.reshape((1000,res*res*3))
NonFTRI=NonFTRI_arr.reshape((1000,res*res*3))
TRI=np.zeros((2000,res*res*3))
TRI[0:1000,]=FTRI
TRI[1000:2000,]=NonFTRI



# Reading Test Face images and Nonface images into array


TEI=os.listdir("res"+str(res)+"by"+str(res)+"/extracted_pics/Test")
print(len(TEI))
print(len(os.listdir("res"+str(res)+"by"+str(res)+"/extracted_pics/Test")))#2000 images


FTI=[name for name in TEI if name.startswith("face_")==True]
NFTI=[name for name in TEI if name.startswith("nonface_")==True]


print(len(FTI))
print(len(NFTI))


FTI_arr=np.zeros((100,res,res,3))
for i in range(100):# 3 doesn'r work
    #print(i)
    #type(i)
    rbg_arr=mpimg.imread("res"+str(res)+"by"+str(res)+"/extracted_pics/Test/"+FTI[i])
    
    if rbg_arr.shape!=(res,res,3):
        print("False")
    FTI_arr[i,:,:,:]=rbg_arr
    
    
NFTI_arr=np.zeros((100,res,res,3))
for i in range(100):# 3 doesn'r work
    #print(i)
    #print(NFTI[i])
    #type(i)
    rbg_arr=mpimg.imread("res"+str(res)+"by"+str(res)+"/extracted_pics/Test/"+NFTI[i])
    
    if rbg_arr.shape!=(res,res,3):
        print("False")
    NFTI_arr[i,:,:,:]=rbg_arr
   

FTI = FTI_arr.reshape((100,res*res*3))
NFTI = NFTI_arr.reshape((100,res*res*3))
TEI = np.zeros((200,res*res*3))
TEI[0:100,] = FTI
TEI[100:200,] = NFTI

