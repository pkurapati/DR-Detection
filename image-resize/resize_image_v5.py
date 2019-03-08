from PIL import Image
import subprocess
import os
import os.path
import cv2 , glob , numpy
import sys


# Check for number of args

if len(sys.argv)!=6:
    print("Error!! Please pass all the arguments\n")
    print("<input dir><output dir><fuzz><size><aug>")
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
    aug = int(sys.argv[5])

    for filename in os.listdir(image_dir_input):
        f_input = image_dir_input + filename
        f_output = image_dir_output + filename
        # Check if we already created the file
        if os.path.isfile(f_output):
            continue
        else:
            # Write 3 images : flip, flop and rotate and shear
            filename = filename.split('.')[0]
            f0 = image_dir_output + filename + '.jpeg'
            cmd0 = 'gm convert %s -fuzz %i%% -trim -scale %ix%i -gravity center -extent %ix%i -quality 100 %s' % (f_input,fuzz,scale,scale,scale,scale,f0)
            subprocess.call(cmd0,shell=True)
            if aug==1:
                f1 = image_dir_output + filename + '-flip.jpeg'
                cmd1 = 'gm convert %s -fuzz %i%% -trim -scale %ix%i -gravity center -extent %ix%i -quality 100 -flip %s' % (f_input,fuzz,scale,scale,scale,scale,f1)
                subprocess.call(cmd1,shell=True)

                f2 = image_dir_output + filename + '-flop.jpeg'
                cmd2 = 'gm convert %s -fuzz %i%% -trim -scale %ix%i -gravity center -extent %ix%i -quality 100 -flop %s' % (f_input,fuzz,scale,scale,scale,scale,f2)
                subprocess.call(cmd2,shell=True)

                f3 = image_dir_output + filename + '-rs.jpeg'
                cmd3 = 'gm convert -rotate 25 -fill black %s -fuzz %i%% -trim -scale %ix%i -gravity center -extent %ix%i -quality 100 -flop %s' % (f_input,fuzz,scale,scale,scale,scale,f3)
                subprocess.call(cmd3,shell=True)

            
            

                
                
                

