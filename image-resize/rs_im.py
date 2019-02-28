from PIL import Image
import subprocess
import os
import os.path
import cv2 , glob , numpy
import sys


# Check for number of args

if len(sys.argv)!=5:
    print("Error!! Please pass all the arguments\n")
    print("<input dir><output dir><fuzz><size>")
else:
    # Image directory where original images are stored
    image_dir_input = os.getcwd() + '/' + sys.argv[1] + '/'

    # Image directory where resized images are stored
    image_dir_output = os.getcwd() + '/' + sys.argv[2] + '/'

    # Create the output directory if it does not exist
    if not os.path.exists(image_dir_output):
        os.makedirs(image_dir_output)

    fuzz = int(sys.argv[3])
    scale = int(sys.argv[4])

    for filename in os.listdir(image_dir_input):
        f_input = image_dir_input + filename
        f_output = image_dir_output + filename
        # Check if we already created the file
        if os.path.isfile(f_output):
            continue
        else:
            cmd = 'gm convert %s -fuzz %i%% -trim -scale %ix%i -gravity center -extent %ix%i -quality 100 %s' % (f_input,fuzz,scale,scale,scale,scale,f_output)
            subprocess.call(cmd,shell=True)

            
            

                
                
                

