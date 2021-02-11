'''*** Import Section ***'''
from __future__ import division              # to allow compatibility of code between Python 2.x and 3.x with minimal overhead
from collections import Counter              # library and method for counting hashable objects
import argparse                              # to define arguments to the program in a user-friendly way
import os                                    # provides functions to interact with local file system
import os.path as osp                        # provides range of methods to manipulate files and directories
import pickle as pkl                         # to implement binary protocols for serializing and de-serializing object structure
import pandas as pd                          # popular data-analysis library for machine learning.
import time                                  # for time-related python functions
import sys                                   # provides access for variables used or maintained by intrepreter
import torch                                 # machine learning library for tensor and neural-network computations
from torch.autograd import Variable          # Auto Differentaion package for managing scalar based values
import cv2                                   # OpenCV Library to carry out Computer Vision tasks
import warnings                              # to manage warnings that are displayed during execution
warnings.filterwarnings('ignore')            # to ignore warning messages while code execution
from util.parser import load_classes         # navigates to load_classess function in util.parser.py
from util.model import Darknet               # to load weights into our model for vehicle detection
from util.image_processor import prep_image  # to pass input image into model,after resizing it into yolo format
from util.utils import non_max_suppression   # to do non-max-suppression in the detected bounding box objects i.e cars


print('\033[1m'+'\033[91m'+"Kickstarting YOLO...\n")

'''*** Parsing Arguments to YOLO Model ***'''
def arg_parse():
    parser = argparse.ArgumentParser(description='YOLO Vehicle Detection Model for Intelligent Traffic Management System')
    parser.add_argument("--images",dest='images',help="Image / Directory containing images to  vehicle detection upon",default="/content/Model/test-images",type=str)
    '''parser.add_argument("--outputs",dest='outputs',help="Image / Directory to store detections",default="/content/output/",type=str)'''
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence_score",dest="confidence",help="Confidence Score to filter Vehicle Prediction",default=0.5)
    parser.add_argument("--nms_thresh",dest="nms_thresh",help="NMS Threshhold",default=0.4)
    parser.add_argument("--cfg",dest='cfgfile',help="Config file",default="config/yolov3.cfg",type=str)
    parser.add_argument("--weights",dest='weightsfile',help="weightsfile",default="weights/yolov3.weights",type=str)
    parser.add_argument("--reso",dest='reso',help="Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",default="416",type=str)
    return parser.parse_args()

args = arg_parse()
images = args.images
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()


'''***Loading Dataset Class File***'''
classes = load_classes("data/idd.names")


'''***Setting up the neural network***'''
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print('\033[0m'+"Input Data Passed Into YOLO Model..."+u'\N{check mark}')
print('\033[0m'+"YOLO Neural Network Successfully Loaded..."+u'\N{check mark}')
print('\033[0m')
model.hyperparams["height"] = args.reso
inp_dim = int(model.hyperparams["height"])
assert inp_dim % 32 == 0
assert inp_dim > 32
num_classes = model.num_classes

#Putting YOLO Model into GPU:
if CUDA:
    model.cuda()
model.eval()
read_dir = time.time()


'''***Vehicle Detection Phase***'''
vehicle_count = 0
try:
    imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
except NotADirectoryError:
    imlist = []
    imlist.append(osp.join(osp.realpath('.'), images))
except FileNotFoundError:
    print("No Input with the name {}".format(images))
    print("Model failed to load your input.  ")
    exit()

'''if not os.path.exists(args.outputs):
    os.makedirs(args.outputs)'''

load_batch = time.time()
loaded_ims = [cv2.imread(x) for x in imlist]

im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(imlist))]))
im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

leftover = 0

if (len(im_dim_list) % batch_size):
    leftover = 1

if batch_size != 1:
    num_batches = len(imlist) // batch_size + leftover
    im_batches = [
        torch.cat((im_batches[i * batch_size:min((i + 1) *batch_size, len(im_batches))]))
        for i in range(num_batches)
    ]

print('\033[1m'+'\033[92m'+ "Performing Vehicle Detection..."+'\033[0m'+u'\N{check mark}')

write = 0

if CUDA:
    im_dim_list = im_dim_list.cuda()
start_outputs_loop = time.time()

print()
print('\033[0m' +"------------------------------------------------------------------------------------------------------------------------------------------------------------")
print('\033[1m' +"SUMMARY")
print('\033[0m' +"------------------------------------------------------------------------------------------------------------------------------------------------------------")
print('\033[1m' + "{:45s}: {}".format("Task", "Time Taken (in seconds)"))
print('\033[0m')


#Loading the image, if present :
for i, batch in enumerate(im_batches):
    #load the image
    start = time.time()
    if CUDA:
        batch = batch.cuda()
    with torch.no_grad():
        prediction = model(Variable(batch))

    prediction = non_max_suppression(prediction,confidence,num_classes,nms_conf=nms_thesh)
    end = time.time()

    if type(prediction) == int:
        for im_num, image in enumerate(
                imlist[i * batch_size:min((i + 1) * batch_size, len(imlist))]):
            im_id = i * batch_size + im_num
            print("{0:20s} predicted in {1:6.3f} seconds".format(
                image.split("/")[-1], (end - start) / batch_size))
            print("{0:20s} {1:s}".format("Objects detected:", ""))
            print("----------------------------------------------------------")
        continue

    prediction[:,0] += i * batch_size  # transform the atribute from index in batch to index in imlist

    if not write:                      # If we have't initialised output
        output = prediction
        write = 1
    else:
        output = torch.cat((output, prediction))

    for im_num, image in enumerate(imlist[i * batch_size:min((i + 1) * batch_size, len(imlist))]):
        im_id = i * batch_size + im_num
        objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
        vc = Counter(objs)
        #print("Input File Name : {0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
        print("{0:45s}: {1:6.3f} seconds".format("Detecting Input File Passed into YOLO Model",(end - start) / batch_size))
        #print("{0:20s} {1:s}".format("Objects detected:\n", "\n".join(objs)))
        for i in objs:
            if i == "car" or i == "motorbike" or i == "truck" or i == "bicycle" or i == "autorickshaw":
                vehicle_count += 1

    if CUDA:
        torch.cuda.synchronize()
    if vehicle_count == 0:
        print('\033[1m' +"There are no vehicles present from the input that was passed into our YOLO Model.")
        
try:
    output
except NameError:
    print("No detections were made | No Objects were found from the input")
    exit()


print("------------------------------------------------------------------------------------------------------------------------------------------------------------")
print('\033[0m' + "{:45s}: {:2.3f}".format("Loading batch",
                               start_outputs_loop - load_batch))
print("{:45s}: {:2.3f}".format("Average time for detecting an image",
                               (end - load_batch) / len(imlist)))
print("------------------------------------------------------------------------------------------------------------------------------------------------------------")
print('\033[0m' + "{:45s}: {}".format("Total Number of Objects Detected",len(objs)))
print('\033[0m' + "{:45s}: {}".format("Types of Objects detected with count",vc.most_common()))
print('\033[0m' +"------------------------------------------------------------------------------------------------------------------------------------------------------------")
print('\033[1m' + "{:45s}: {}".format("Number of Vehicles detected in total", vehicle_count))
print('\033[0m'+"------------------------------------------------------------------------------------------------------------------------------------------------------------")
print('\033[1m' + "{:15} {}".format("Vehicle Type", "Count"))
print()
for key,value in sorted(vc.items()):
  if key=="car" or key=="motorbike" or key=="truck" or key=="bicycle":
    print('\033[0m' +"{:15s} {}".format(key,value))
torch.cuda.empty_cache()