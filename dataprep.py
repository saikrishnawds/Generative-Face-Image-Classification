

import os
#import numpy as np
import random
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
from PIL import Image

# here we need to ensure that all files are in same directory 'Project 1' as specified in the very end of  the path below:
os.getcwd()


os.chdir("Sai's stuff/Documents/ncsu/Semester-2/Computer Vision/Project 1")


res=15


ind=["01","02","03","04","05"]
facetotcount=0


cind=ind[0]
for cind in ind[0:3]:
    file_annotation=open("FDDB-folds/FDDB-fold-"+cind+"-ellipseList.txt")
    for line in file_annotation:
        #line=file_annotation.readline()
        img_name = line.rstrip()
        print(img_name)
        img_face_count = int(file_annotation.readline().rstrip())
        print(img_face_count)
        img_file = Image.open("originalPics/"+img_name+".jpg").convert('RGB')
        plt.imshow(img_file)
        #extract face images
        for _ in range(img_face_count):
            temp_range=file_annotation.readline().rstrip().split(" ")
            (range1,range2,angle,x_center,y_center)=([float(i) for i in temp_range[0:5]])
            sq_len=max(range1,range2)
            #print(sq_len)
            area=(x_center-sq_len,y_center-sq_len,x_center+sq_len,y_center+sq_len)
            cropped_img = img_file.crop(area).resize((res,res),Image.ANTIALIAS)  # size
            plt.imshow(cropped_img)
            #resize is to change to exact size
            #do not use cropped_img.thumbnail(size, Image.ANTIALIAS), it change to max size not exact
            facetotcount=facetotcount+1
            cropped_img.save("res"+str(res)+"by"+str(res)+"/extracted_face_pics/"+cind+"/face_"+str(facetotcount)+".jpg")
            #check whether 60,60,60,3
  
        img_file.close()
    file_annotation.close()


        
nonfacetotcount=0
for cind in ind[0:3]:
    file_annotation=open("FDDB-folds/FDDB-fold-"+cind+"-ellipseList.txt")
    for line in file_annotation:
        img_name = line.rstrip()
        print(img_name)
        img_face_count = int(file_annotation.readline().rstrip())
        print(img_face_count)
        img_file = Image.open("originalPics/"+img_name+".jpg") 
        plt.imshow(img_file)
        #extract face images
        for _ in range(img_face_count):
            temp_range=file_annotation.readline()
            x_center=random.uniform(1, img_file.size[0]/4
            y_center=random.uniform(1, img_file.size[1]/4)
            sq_len=random.uniform(1,min(x_center,y_center))
            cropped_img=img_file.crop((x_center-sq_len,y_center-sq_len,x_center+sq_len,y_center+sq_len)).resize((res,res),Image.ANTIALIAS) 
            plt.imshow(cropped_img)
            nonfacetotcount=nonfacetotcount+1
            cropped_img.save("res"+str(res)+"by"+str(res)+"/extracted_nonface_pics/"+cind+"/nonface_"+str(nonfacetotcount)+".jpg")
        
        img_file.close()
    file_annotation.close()
        




print(len(os.listdir("res"+str(res)+"by"+str(res)+"/extracted_face_pics/All")))
print(len(os.listdir("res"+str(res)+"by"+str(res)+"/extracted_nonface_pics/All")))




os.system("pwd")
os.system("cp res"+str(res)+"by"+str(res)+"/extracted_face_pics/01/*.jpg res"+ \
          str(res)+"by"+str(res)+"/extracted_face_pics/All")


os.system("cp res"+str(res)+"by"+str(res)+"/extracted_face_pics/02/*.jpg res"+ \
          str(res)+"by"+str(res)+"/extracted_face_pics/All")


os.system("cp res"+str(res)+"by"+str(res)+"/extracted_face_pics/03/*.jpg res"+ \
          str(res)+"by"+str(res)+"/extracted_face_pics/All")


os.system("cp res"+str(res)+"by"+str(res)+"/extracted_nonface_pics/01/*.jpg res"+ \
          str(res)+"by"+str(res)+"/extracted_nonface_pics/All")


os.system("cp res"+str(res)+"by"+str(res)+"/extracted_nonface_pics/02/*.jpg res"+ \
          str(res)+"by"+str(res)+"/extracted_nonface_pics/All")


os.system("cp res"+str(res)+"by"+str(res)+"/extracted_nonface_pics/03/*.jpg res"+ \
          str(res)+"by"+str(res)+"/extracted_nonface_pics/All")




face_img=os.listdir("res"+str(res)+"by"+str(res)+"/extracted_face_pics/All")
num_face_img=len(face_img)#1551
nonface_img=os.listdir("res"+str(res)+"by"+str(res)+"/extracted_nonface_pics/All")
num_nonface_img=len(nonface_img)
print(num_face_img)
print(num_nonface_img)




        
face_train_ind = random.sample(range(0,num_face_img),1000)
face_test_ind = random.sample([i for i in range(0,num_face_img) if not i in face_train_ind],100)
nonface_train_ind=random.sample(range(0,num_nonface_img),1000)
nonface_test_ind = random.sample([i for i in range(0,num_nonface_img) if not i in nonface_train_ind],100)




#face images training
for ind in face_train_ind:
    os.system("cp res"+str(res)+"by"+str(res)+"/extracted_face_pics/All/face_"+str(ind+1)+".jpg res"+str(res)+"by"+str(res)+"/extracted_face_pics/Train")


#face images testing
for ind in face_test_ind:
    os.system("cp res"+str(res)+"by"+str(res)+"/extracted_face_pics/All/face_"+str(ind+1)+".jpg res"+str(res)+"by"+str(res)+"/extracted_face_pics/Test")


#nonface images training
for ind in nonface_train_ind:
    os.system("cp res"+str(res)+"by"+str(res)+"/extracted_nonface_pics/All/nonface_"+str(ind+1)+".jpg res"+str(res)+"by"+str(res)+"/extracted_nonface_pics/Train")


#nonface images testing
for ind in nonface_test_ind:
    os.system("cp res"+str(res)+"by"+str(res)+"/extracted_nonface_pics/All/nonface_"+str(ind+1)+".jpg res"+str(res)+"by"+str(res)+"/extracted_nonface_pics/Test")




print(len(os.listdir("res"+str(res)+"by"+str(res)+"/extracted_face_pics/Train")))
print(len(os.listdir("res"+str(res)+"by"+str(res)+"/extracted_face_pics/Test")))
print(len(os.listdir("res"+str(res)+"by"+str(res)+"/extracted_nonface_pics/Train")))
print(len(os.listdir("res"+str(res)+"by"+str(res)+"/extracted_nonface_pics/Test")))


os.system("cp res"+str(res)+"by"+str(res)+"/extracted_face_pics/Train/*.jpg res"+str(res)+"by"+str(res)+"/extracted_pics/Train")
os.system("cp res"+str(res)+"by"+str(res)+"/extracted_nonface_pics/Train/*.jpg res"+str(res)+"by"+str(res)+"/extracted_pics/Train")
os.system("cp res"+str(res)+"by"+str(res)+"/extracted_face_pics/Test/*.jpg res"+str(res)+"by"+str(res)+"/extracted_pics/Test")
os.system("cp res"+str(res)+"by"+str(res)+"/extracted_nonface_pics/Test/*.jpg res"+str(res)+"by"+str(res)+"/extracted_pics/Test")


print(len(os.listdir("res"+str(res)+"by"+str(res)+"/extracted_pics/Train")))
print(len(os.listdir("res"+str(res)+"by"+str(res)+"/extracted_pics/Test")))

