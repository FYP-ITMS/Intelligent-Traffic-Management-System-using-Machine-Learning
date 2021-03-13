#!/usr/bin/env python

from collections import defaultdict

import numpy as np
import torch
from torch import nn

from .parser import parse_model_configuration
from .moduler import modules_creator

class Darknet(nn.Module):
    """YOLOv3 object detection model"""
    def __init__(self, config_path, img_size=416):
        """
        Function:
            Constructor for Darknet class
    
        Arguments:
            conig_path -- path of model configuration file
            img_size -- size of input images
        """
        super(Darknet, self).__init__()
        self.blocks = parse_model_configuration(config_path)
        self.hyperparams, self.module_list ,self.num_classes = modules_creator(self.blocks)
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0])
        self.loss_names = ["x", "y", "w", "h", "conf", "cls", "recall", "precision"]

    def forward(self, x, targets=None):
        """
        Function:
            Feedforward propagation for prediction
        
        Arguments:
            x -- input tensor 
            targets -- tensor of true values of training process
            
        Returns:
            output -- tensor of outputs of the models
        """
        is_training = targets is not None
        output = []
        self.losses = defaultdict(float)
        layer_outputs = []
        for i, (block, module) in enumerate(zip(self.blocks, self.module_list)):
            if block["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif block["type"] == "route":
                layer_i = [int(x) for x in block["layers"].split(",")]
                x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif block["type"] == "shortcut":
                layer_i = int(block["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif block["type"] == "yolo":
                # Train phase: get loss
                if is_training:
                    x, *losses = module[0](x, targets)
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss
                # Test phase: Get detections
                else:
                    x = module(x)
                output.append(x)
            layer_outputs.append(x)

        self.losses["recall"] /= 3
        self.losses["precision"] /= 3
        return sum(output) if is_training else torch.cat(output, 1)

    def load_weights(self, weights_path):        
        """
        Function:
            Parses and loads the weights stored in 'weights_path
            
        Arguments:
            weights_path -- path of weights file
        """
        # Open the weights file
        fp = open(weights_path, "rb")
        header = np.fromfile(fp, dtype=np.int32, count=5)  # First five are header values

        # Needed to write header when saving weights
        self.header_info = header

        self.seen = header[3]
        weights = np.fromfile(fp, dtype=np.float32)  # The rest are weights
        fp.close()

        ptr = 0
        for i, (block, module) in enumerate(zip(self.blocks, self.module_list)):
            if block["type"] == "convolutional":
                conv_layer = module[0]
                try:
                    block["batch_normalize"]
                except:
                    block["batch_normalize"] = 0
                if block["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_weights(self, path, cutoff=-1):
        """
        Function:
            Save trained model's weights
        
        Arguments:
            path -- path of the new weights file
            cutoff -- save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (block, module) in enumerate(zip(self.blocks[:cutoff], self.module_list[:cutoff])):
            if block["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if block["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)
        fp.close()