from PIL import Image
from resizeimage import resizeimage
import os
import cv2 , glob , numpy
from PIL import Image
import subprocess
import os.path
import sys

# Image directory where original images are stored
image_dir_input = os.getcwd() + '/train/'

# Image directory where resized images are stored
image_dir_output = os.getcwd() + '/train_resize_128_new/'

# Create the output directory if it does not exist
if not os.path.exists(image_dir_output):
    os.makedirs(image_dir_output)

def scaleRadius(img,scale):
    x=img[int(img.shape[0]/2),:,:].sum(1)
    r=(x>x.mean()/10).sum()/2
    if r!=0:
        s=scale*1.0/r
        image_new = cv2.resize(img,(0,0),fx=s,fy=s)
    else:
        image_new = None
    return image_new

scale=128

for filename in os.listdir(image_dir_input):
    f = image_dir_input + filename
    new_file = image_dir_output+str(scale)+'_'+filename
    if os.path.isfile(new_file):
        continue
    else:
        a=cv2.imread(f)
        #Scale image to a given Radius
        a=scaleRadius(a,scale)
        if a!=None:
            #Subtract local mean color
            a=cv2.addWeighted(a,4,
                              cv2.GaussianBlur(a,(0,0),scale/30),
                             -4,128)
            #Remove outer 10%
            b=numpy.zeros(a.shape)
            cv2.circle(b,(int(a.shape[1]/2),int(a.shape[0]/2)),
                       int(scale*0.9),(1,1,1),-1,8,0)
            a=a*b+128*(1-b)
            cv2.imwrite(new_file,a)
            print("Written image ",new_file)
        #cv2.imwrite(str(scale)+"_"+f,a)

            
            

                
                
                

