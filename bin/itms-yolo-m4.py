'''*** Import Section ***'''
import IPython
from google.colab import output
from __future__ import division  # to allow compatibility of code between Python 2.x and 3.x with minimal overhead
from collections import Counter  # library and method for counting hashable objects
import argparse  # to define arguments to the program in a user-friendly way
import os  # provides functions to interact with local file system
import os.path as osp  # provides range of methods to manipulate files and directories
import pickle as pkl  # to implement binary protocols for serializing and de-serializing object structure
import pandas as pd  # popular data-analysis library for machine learning.
import time  # for time-related python functions
import sys  # provides access for variables used or maintained by intrepreter
import torch  # machine learning library for tensor and neural-network computations
from torch.autograd import Variable  # Auto Differentaion package for managing scalar based values
import cv2  # OpenCV Library to carry out Computer Vision tasks
import emoji
import warnings  # to manage warnings that are displayed during execution
warnings.filterwarnings(
    'ignore')  # to ignore warning messages while code execution
print('\033[1m' + '\033[91m' + "Kickstarting YOLO...\n")
from util.parser import load_classes  # navigates to load_classess function in util.parser.py
from util.model import Darknet  # to load weights into our model for vehicle detection
from util.image_processor import preparing_image  # to pass input image into model,after resizing it into yolo format
from util.utils import non_max_suppression  # to do non-max-suppression in the detected bounding box objects i.e cars


#*** Parsing Arguments to YOLO Model ***
def arg_parse():
    parser = argparse.ArgumentParser(
        description=
        'YOLO Vehicle Detection Model for Intelligent Traffic Management System'
    )
    parser.add_argument(
        "--images",
        dest='images',
        help="Image / Directory containing images to  vehicle detection upon",
        default="/content/Model/test-images",
        type=str)
    '''parser.add_argument("--outputs",dest='outputs',help="Image / Directory to store detections",default="/content/output/",type=str)'''
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence_score",
                        dest="confidence",
                        help="Confidence Score to filter Vehicle Prediction",
                        default=0.3)
    parser.add_argument("--nms_thresh",
                        dest="nms_thresh",
                        help="NMS Threshhold",
                        default=0.3)
    parser.add_argument("--cfg",
                        dest='cfgfile',
                        help="Config file",
                        default="config/yolov3.cfg",
                        type=str)
    parser.add_argument("--weights",
                        dest='weightsfile',
                        help="weightsfile",
                        default="weights/yolov3.weights",
                        type=str)
    parser.add_argument(
        "--reso",
        dest='reso',
        help=
        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
        default="416",
        type=str)
    return parser.parse_args()


args = arg_parse()
images = args.images
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()

#***Loading Dataset Class File***
classes = load_classes("data/idd.names")

#***Setting up the neural network***
model = Darknet(args.cfgfile)
print('\033[0m' + "Input Data Passed Into YOLO Model..." + u'\N{check mark}')
model.load_weights(args.weightsfile)
print('\033[0m' + "YOLO Neural Network Successfully Loaded..." +
      u'\N{check mark}')
print('\033[0m')
model.hyperparams["height"] = args.reso
inp_dim = int(model.hyperparams["height"])
assert inp_dim % 32 == 0
assert inp_dim > 32
num_classes = model.num_classes
print('\033[1m' + '\033[94m' +
      "Performing Vehicle Detection with YOLO Neural Network..." + '\033[0m' +
      u'\N{check mark}')
#Putting YOLO Model into GPU:
if CUDA:
    model.cuda()
model.eval()
read_dir = time.time()

#***Vehicle Detection Phase***
try:
    imlist = [
        osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)
    ]
except NotADirectoryError:
    imlist = []
    imlist.append(osp.join(osp.realpath('.'), images))
except FileNotFoundError:
    print("No Input with the name {}".format(images))
    print("Model failed to load your input.  ")
    exit()

load_batch = time.time()
loaded_ims = [cv2.imread(x) for x in imlist]

im_batches = list(
    map(preparing_image, loaded_ims, [inp_dim for x in range(len(imlist))]))
im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

leftover = 0

if (len(im_dim_list) % batch_size):
    leftover = 1

if batch_size != 1:
    num_batches = len(imlist) // batch_size + leftover
    im_batches = [
        torch.cat(
            (im_batches[i * batch_size:min((i + 1) *
                                           batch_size, len(im_batches))]))
        for i in range(num_batches)
    ]

write = 0

if CUDA:
    im_dim_list = im_dim_list.cuda()
start_outputs_loop = time.time()

input_image_count = 0
denser_lane = 0
lane_with_higher_count = 0
print()
print(
    '\033[1m' +
    "------------------------------------------------------------------------------------------------------------------------------------------------------------"
)
print('\033[1m' + "SUMMARY")
print(
    '\033[1m' +
    "------------------------------------------------------------------------------------------------------------------------------------------------------------"
)
print('\033[1m' +
      "{:25s}: ".format("Detected  (" + str(len(imlist)) + " inputs)"))
print('\033[0m')
#Loading the image, if present :
for i, batch in enumerate(im_batches):
    #load the image
    vehicle_count = 0
    start = time.time()
    if CUDA:
        batch = batch.cuda()
    with torch.no_grad():
        prediction = model(Variable(batch))

    prediction = non_max_suppression(prediction,
                                     confidence,
                                     num_classes,
                                     nms_conf=nms_thesh)

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

    prediction[:,
               0] += i * batch_size  # transform the atribute from index in batch to index in imlist

    if not write:  # If we have't initialised output
        output = prediction
        write = 1
    else:
        output = torch.cat((output, prediction))

    for im_num, image in enumerate(
            imlist[i * batch_size:min((i + 1) * batch_size, len(imlist))]):
        vehicle_count = 0
        input_image_count += 1
        #denser_lane =
        im_id = i * batch_size + im_num
        objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
        vc = Counter(objs)
        for i in objs:
            if i == "car" or i == "motorbike" or i == "truck" or i == "bicycle" or i == "autorickshaw":
                vehicle_count += 1

        print('\033[1m' + "Lane : {} - {} : {:5s} {}".format(
            input_image_count, "Number of Vehicles detected", "",
            vehicle_count))
        if vehicle_count > lane_with_higher_count:
            lane_with_higher_count = vehicle_count
            denser_lane = input_image_count
        print(
            '\033[0m' +
            "           File Name:     {0:20s}.".format(image.split("/")[-1]))
        print('\033[0m' +
              "           {:15} {}".format("Vehicle Type", "Count"))
        for key, value in sorted(vc.items()):
            if key == "car" or key == "motorbike" or key == "truck" or key == "bicycle":
                print('\033[0m' + "            {:15s} {}".format(key, value))

    if CUDA:
        torch.cuda.synchronize()
    if vehicle_count == 0:
        print(
            '\033[1m' +
            "There are no vehicles present from the input that was passed into our YOLO Model."
        )

print(
    '\033[1m' +
    "----------------------------------------------------------------------" +
    "\n")
print(
    emoji.emojize(':vertical_traffic_light:') + '\033[1m' + '\033[92m' +
    "  Lane with denser traffic is :" + str(denser_lane) + "\n")

try:
    output
except NameError:
    print("No detections were made | No Objects were found from the input")
    exit()

display(
    IPython.display.HTML('''<div class="container">
	<div class="circle red" color="red"></div>
	<div class="circle" color="yellow"></div>
	<div class="circle" color="green"></div>



<div class="floating-text">
	Intelligent Traffic Management System </div>

<style>
@import url('https://fonts.googleapis.com/css?family=Muli&display=swap');

* {
	box-sizing: border-box;
}

body {
	background-color: #1abc9c;
	display: flex;
	align-items: center;
	justify-content: center;
	min-height: 100vh;
	margin: 0;
}

.container {
	background-color: #2c3e50;
	border-radius: 50px;
	display: flex;
	flex-direction: column;
	align-items: center;
	justify-content: space-around;
	padding: 15px 0;
	height: 200px;
	width: 70px;
}

.circle {
	background-color: rgba(0, 0, 0, 0.3);
	border-radius: 100%;
	position: relative;
	height: 40px;
	width: 40px;
}

.circle::after {
	border-right: 4px solid rgba(255, 255, 255, 0.6);
	border-radius: 100%;
	content: ' ';
	position: absolute;
	top: 5px;
	left: 0px;
	width: 30px;
	height: 30px;
}

.circle.red {
	background-color: #c0392b;
	box-shadow: 0 0 20px 5px #c0392b;
}

.circle.yellow {
	background-color: #f1c40f;
	box-shadow: 0 0 20px 5px #f1c40f;
}

.circle.green {
	background-color: #2ecc71;
	box-shadow: 0 0 20px 5px #2ecc71;
}






/* SOCIAL PANEL CSS */
.social-panel-container {
	position: fixed;
	right: 0;
	bottom: 80px;
	transform: translateX(100%);
	transition: transform 0.4s ease-in-out;
}

.social-panel-container.visible {
	transform: translateX(-10px);
}

.social-panel {	
	background-color: #fff;
	border-radius: 16px;
	box-shadow: 0 16px 31px -17px rgba(0,31,97,0.6);
	border: 5px solid #001F61;
	display: flex;
	flex-direction: column;
	justify-content: center;
	align-items: center;
	font-family: 'Muli';
	position: relative;
	height: 169px;	
	width: 370px;
	max-width: calc(100% - 10px);
}

.social-panel button.close-btn {
	border: 0;
	color: #97A5CE;
	cursor: pointer;
	font-size: 20px;
	position: absolute;
	top: 5px;
	right: 5px;
}

.social-panel button.close-btn:focus {
	outline: none;
}

.social-panel p {
	background-color: #001F61;
	border-radius: 0 0 10px 10px;
	color: #fff;
	font-size: 14px;
	line-height: 18px;
	padding: 2px 17px 6px;
	position: absolute;
	top: 0;
	left: 50%;
	margin: 0;
	transform: translateX(-50%);
	text-align: center;
	width: 235px;
}

.social-panel p i {
	margin: 0 5px;
}

.social-panel p a {
	color: #FF7500;
	text-decoration: none;
}

.social-panel h4 {
	margin: 20px 0;
	color: #97A5CE;	
	font-family: 'Muli';	
	font-size: 14px;	
	line-height: 18px;
	text-transform: uppercase;
}

.social-panel ul {
	display: flex;
	list-style-type: none;
	padding: 0;
	margin: 0;
}

.social-panel ul li {
	margin: 0 10px;
}

.social-panel ul li a {
	border: 1px solid #DCE1F2;
	border-radius: 50%;
	color: #001F61;
	font-size: 20px;
	display: flex;
	justify-content: center;
	align-items: center;
	height: 50px;
	width: 50px;
	text-decoration: none;
}

.social-panel ul li a:hover {
	border-color: #FF6A00;
	box-shadow: 0 9px 12px -9px #FF6A00;
}

.floating-btn {
	border-radius: 26.5px;
	background-color: #001F61;
	border: 1px solid #001F61;
	box-shadow: 0 16px 22px -17px #03153B;
	color: #fff;
	cursor: pointer;
	font-size: 16px;
	line-height: 20px;
	padding: 12px 20px;
	position: fixed;
	bottom: 20px;
	right: 20px;
	z-index: 999;
}

.floating-btn:hover {
	background-color: #ffffff;
	color: #001F61;
}

.floating-btn:focus {
	outline: none;
}

.floating-text {
	background-color: #001F61;
	border-radius: 10px 10px 0 0;
	color: #fff;
	font-family: 'Muli';
	padding: 7px 15px;
	position: fixed;
	bottom: 0;
	left: 50%;
	transform: translateX(-50%);
	text-align: center;
	z-index: 998;
}

.floating-text a {
	color: #FF7500;
	text-decoration: none;
}

@media screen and (max-width: 480px) {

	.social-panel-container.visible {
		transform: translateX(0px);
	}
	
	.floating-btn {
		right: 10px;
	}
}
</style>
<script>
const circles = document.querySelectorAll('.circle')
let activeLight = 0;

setInterval(() => {
	changeLight();
}, 1000);

function changeLight() {
	circles[activeLight].className = 'circle';
	activeLight++;
	
	if(activeLight > 2) {
		activeLight = 0;
	}
	
	const currentLight = circles[activeLight]
	
	currentLight.classList.add(currentLight.getAttribute('color'));
}

</script>
    '''))

torch.cuda.empty_cache()