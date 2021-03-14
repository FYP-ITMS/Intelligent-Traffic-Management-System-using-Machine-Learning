import math
import numpy as np
import torch

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Function:
        Calculate intersection over union between two bboxes
        
    Arguments:
       box1 -- first bbox 
       box2 -- second bbox
       x1y1x2y2 -- bool value
    Return:
        iou --  the IoU of two bounding boxes 
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def unique(tensor):
    """
    Function:
        Get the various classes detected in the image
    
    Arguments:
        tensor -- torch tensor
    
    Return:
        tensor_res -- torch tensor after preparing 
    """
    tensor_np = tensor.detach().cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res




def non_max_suppression(prediction, confidence, num_classes, nms_conf = 0.4):
    """
    Function:
        Removes detections with lower object confidence score than 'conf_thres' and performs
        Non-Maximum Suppression to further filter detections.
        
    Arguments:
        prediction -- tensor of yolo model prediction
        confidence -- float value to remove all prediction has confidence value low than the confidence
        num_classes -- number of class
        nms_conf -- float value (non max suppression) to remove bbox it's iou larger than nms_conf
    
    Return:
        output -- tuple (x1, y1, x2, y2, object_conf, class_score, class_pred) 
    """
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask
    
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]
    
    batch_size = prediction.size(0)

    write = False
    


    for ind in range(batch_size):
        image_pred = prediction[ind]          #image Tensor
       #confidence threshholding 
       #NMS
    
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)
        
        non_zero_ind =  (torch.nonzero(image_pred[:,4]))
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        except:
            continue
        
        if image_pred_.shape[0] == 0:
            continue       
#        
  
        #Get the various classes detected in the image
        img_classes = unique(image_pred_[:,-1])  # -1 index holds the class index
        
        
        for cls in img_classes:
            #perform NMS

        
            #get the detections with one particular class
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1,7)
            
            #sort the detections such that the entry with the maximum objectness
            #confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)   #Number of detections
            
            for i in range(idx):
                #Get the IOUs of all boxes that come after the one we are looking at 
                #in the loop
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except ValueError:
                    break
            
                except IndexError:
                    break
            
                #Zero out all the detections that have IoU > treshhold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask       
            
                #Remove the non-zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)
                
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)      #Repeat the batch_id for as many detections of the class cls in the image
            seq = batch_ind, image_pred_class
            
            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))

    try:
        return output
    except:
        return 0


def build_targets(pred_boxes, pred_conf, pred_cls, target, anchors, num_anchors,
                  num_classes, grid_size, ignore_thres, img_dim):
    """
    Function:
        build the target values for training process        
    Arguments:
       pred_boxes -- predicted bboxes
       pred_conf -- predicted confidence of bbox
       pred_cls -- predicted class of bbox 
       target -- target value
       anchors -- list of anchors boxs' dimensions 
       num_anchors -- number of anchor boxes
       num_classes -- number of classes
       grid_size -- grid size
       ignore_thres -- confidence thres
       img_dim -- input image dimension
    
    Return:
        nGT -- total number of predictions
        n_correct -- number of correct predictions
        mask -- mask
        conf_mask -- confidence mask
        tx -- xs of bboxes
        ty -- ys of bboxs
        tw -- width of bbox
        th -- height of bbox
        tconf -- confidence
        tcls  -- class prediction
    """

    
    batch_size = target.size(0)
    num_anchors = num_anchors
    num_classes = num_classes
    n_grid = grid_size
    
    mask = torch.zeros(batch_size, num_anchors, n_grid, n_grid)
    conf_mask = torch.ones(batch_size, num_anchors, n_grid, n_grid)
    
    tx = torch.zeros(batch_size, num_anchors, n_grid, n_grid)
    ty = torch.zeros(batch_size, num_anchors, n_grid, n_grid)
    tw = torch.zeros(batch_size, num_anchors, n_grid, n_grid)
    th = torch.zeros(batch_size, num_anchors, n_grid, n_grid)

    tconf = torch.ByteTensor(batch_size, num_anchors, n_grid, n_grid).fill_(0)
    tcls = torch.ByteTensor(batch_size, num_anchors, n_grid, n_grid, num_classes).fill_(0)

    nGT = 0
    n_correct = 0
    
    for b in range(batch_size):
        for t in range(target.shape[1]):
            if target[b, t].sum == 0:
                continue
            
            nGT += 1
            
            # Convert to position relative to box
            gx = target[b, t, 1] * n_grid
            gy = target[b, t, 2] * n_grid
            gw = target[b, t, 3] * n_grid
            gh = target[b, t, 4] * n_grid
            
            # Get grid box indices
            gi = int(gx)
            gj = int(gy)
            
            # Get shape of gt box
            gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
            
            # Get shape of anchor box
            anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(anchors), 2)), np.array(anchors)), 1))
            
            # Calculate iou between gt and anchor shapes
            anch_ious = bbox_iou(gt_box, anchor_shapes)
            
            # Where the overlap is larger than threshold set mask to zero (ignore)
            conf_mask[b, anch_ious > ignore_thres, gj, gi] = 0
            
            # Find the best matching anchor box
            best_n = np.argmax(anch_ious)
            
            # Get ground truth box
            gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0)
            
            # Get the best prediction
            pred_box = pred_boxes[b, best_n, gj, gi].unsqueeze(0)

            # Masks
            mask[b, best_n, gj, gi] = 1
            conf_mask[b, best_n, gj, gi] = 1
            
            # Coordinates
            tx[b, best_n, gj, gi] = gx - gi
            ty[b, best_n, gj, gi] = gy - gj
            
            # Width and height
            tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
            th[b, best_n, gj, gi] = math.log(gh / anchors[best_n][1] + 1e-16)
            
            # One-hot encoding of label
            target_label = int(target[b, t, 0])
            tcls[b, best_n, gj, gi, target_label] = 1
            tconf[b, best_n, gj, gi] = 1

            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
            pred_label = torch.argmax(pred_cls[b, best_n, gj, gi])
            score = pred_conf[b, best_n, gj, gi]
            if iou > 0.5 and pred_label == target_label and score > 0.5:
                n_correct += 1

    return nGT, n_correct, mask, conf_mask, tx, ty, tw, th, tconf, tcls
            


def weights_init_normal(m):
    """
    Function:
        Initialize weights
        
    Arguments:
        m -- module
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

