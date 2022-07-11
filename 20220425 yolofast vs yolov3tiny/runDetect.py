#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation.
# License: MIT See LICENSE file in root directory.

from timeit import default_timer as timer
import cv2
import numpy as np, math
import sys
import time, yaml
# DETECTION_THRESHOLD = 0.4
IOU_THRESHOLD = 0.30
crop_w = 400
crop_h = 400

#parameter
with open('./thermalDetect/class_parameters.yml') as file:
    parameters = yaml.load(file, Loader=yaml.FullLoader)
    # print(parameters)
    classes_para = parameters["classes"]

#---NCS2
num = 3 # achor number set at one time
coords = 4 # xywh
classes = classes_para #3 #8run_inference1_preprocess time = 0.011801  sec
anchors = [10,14, 23,27, 37,58, 81,82, 135,169, 344,319]
new_w = 416
new_h = 416
m_input_size = 416
yolo_scale_13 = 13
yolo_scale_26 = 26
yolo_scale_52 = 52
#---

camera_width = 640
camera_height = 480

def preprocess_image(image):
    global new_w, new_h
    if(new_w == 0 or new_h == 0):
        new_w = int(camera_width * min(m_input_size/camera_width, m_input_size/camera_height))
        new_h = int(camera_height * min(m_input_size/camera_width, m_input_size/camera_height))
        print("new w/h: {}/{}".format(new_w, new_h))

    resized_image = cv2.resize(image, (new_w, new_h), interpolation = cv2.INTER_CUBIC)
    canvas = np.full((m_input_size, m_input_size, 3), 128)
    canvas[(m_input_size-new_h)//2:(m_input_size-new_h)//2 + new_h,(m_input_size-new_w)//2:(m_input_size-new_w)//2 + new_w,  :] = resized_image
    prepimg = canvas
    prepimg = prepimg[np.newaxis, :, :, :]     # Batch size axis add
    prepimg = prepimg.transpose((0, 3, 1, 2))  # NHWC to NCHW
    return prepimg

class DetectionObject():
    xmin = 0
    ymin = 0
    xmax = 0
    ymax = 0
    class_id = 0
    confidence = 0.0

    def __init__(self, x, y, h, w, class_id, confidence, h_scale, w_scale):
        self.xmin = int((x - w / 2) * w_scale)
        self.ymin = int((y - h / 2) * h_scale)
        self.xmax = int(self.xmin + w * w_scale)
        self.ymax = int(self.ymin + h * h_scale)
        self.class_id = class_id
        self.confidence = confidence


def IntersectionOverUnion(box_1, box_2):
    width_of_overlap_area = min(box_1.xmax, box_2.xmax) - max(box_1.xmin, box_2.xmin)
    height_of_overlap_area = min(box_1.ymax, box_2.ymax) - max(box_1.ymin, box_2.ymin)
    area_of_overlap = 0.0
    if (width_of_overlap_area < 0.0 or height_of_overlap_area < 0.0):
        area_of_overlap = 0.0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area
    box_1_area = (box_1.ymax - box_1.ymin)  * (box_1.xmax - box_1.xmin)
    box_2_area = (box_2.ymax - box_2.ymin)  * (box_2.xmax - box_2.xmin)
    area_of_union = box_1_area + box_2_area - area_of_overlap
    retval = 0.0
    if area_of_union <= 0.0:
        retval = 0.0
    else:
        retval = (area_of_overlap / area_of_union)
    return retval

def EntryIndex(side, lcoords, lclasses, location, entry):
             # ( 13,       4,        3, location,     4)
    n = int(location / (side * side))
    loc = location % (side * side)
    return int(n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc)

def ParseYOLOV3Output(blob, new_h, new_w, original_im_h, original_im_w, threshold, objects):
    # ({key1:array(1,24,26,26) / key2:array(1,24,13,13)}, 416, 416, 480, 640, 0.4, initial is [] / second is ...)
    # side 由 blob 所產生的

    out_blob_h = blob.shape[2] # 26 / 13
    out_blob_w = blob.shape[3] # 26 / 13

    side = out_blob_h # 26 / 13
    anchor_offset = 0

    if len(anchors) == 18:   ## YoloV3
        if side == yolo_scale_13:
            anchor_offset = 2 * 6
        elif side == yolo_scale_26:
            anchor_offset = 2 * 3
        elif side == yolo_scale_52:
            anchor_offset = 2 * 0

    elif len(anchors) == 12: ## tiny-YoloV3 #anchors = [10,14, 23,27, 37,58, 81,82, 135,169, 344,319]
        if side == yolo_scale_13: #yolo_scale_13 = 13
            anchor_offset = 2 * 3
        elif side == yolo_scale_26: #yolo_scale_26 = 26
            anchor_offset = 2 * 0

    else:                    ## ???
        if side == yolo_scale_13:
            anchor_offset = 2 * 6
        elif side == yolo_scale_26:
            anchor_offset = 2 * 3
        elif side == yolo_scale_52:
            anchor_offset = 2 * 0

    side_square = side * side
    output_blob = blob.flatten() # output_blob 唯一的輸入變量

    # 預測 bbox 位置
    for i in range(side_square):
        row = int(i / side)
        col = int(i % side)
        for n in range(num): #num = 3  # achor number set at one time
            obj_index = EntryIndex(side, coords, classes, n * side * side + i, coords) #coords = 4 # xywh #classes = 3 #8run_inference1_preprocess time = 0.011801  sec
            box_index = EntryIndex(side, coords, classes, n * side * side + i, 0)
            scale = output_blob[obj_index]
            if (scale < threshold):
                continue
            # tx、ty、tw、th：模型的預測輸出 / cx,cy：grid cell座標 /
            x = (col + output_blob[box_index + 0 * side_square]) / side * new_w # bx = σ(tx)+cx
            y = (row + output_blob[box_index + 1 * side_square]) / side * new_h # by = σ(ty)+cy
            height = math.exp(output_blob[box_index + 3 * side_square]) * anchors[anchor_offset + 2 * n + 1]
            width = math.exp(output_blob[box_index + 2 * side_square]) * anchors[anchor_offset + 2 * n]
            for j in range(classes):
                # j 什麼動作都有
                class_index = EntryIndex(side, coords, classes, n * side_square + i, coords + 1 + j)
                # 唯一由偵測而得來的變量就只有side
                prob = scale * output_blob[class_index]
                # print(output_blob[class_index])
                if prob < threshold:
                    continue
                # j 會被過濾掉
                #obj = DetectionObject(x, y, height, width, j, prob, (original_im_h / resized_im_h), (original_im_w / resized_im_w))
                obj = DetectionObject(x, y, height, width, j, prob, (original_im_h / new_h), (original_im_w / new_w))
                objects.append(obj)
    return objects


def post_proceesing(outputs, image, DETECTION_THRESHOLD_quality_decides):
    DETECTION_THRESHOLD = DETECTION_THRESHOLD_quality_decides
    objects = []
    for output in outputs.values():
        objects = ParseYOLOV3Output(output, new_h, new_w, camera_height, camera_width, DETECTION_THRESHOLD, objects)


    # Filtering overlapping boxes
    # make confidence of overlapping boxes to zero such that we can filter it later
    objlen = len(objects)
    for i in range(objlen):
        if (objects[i].confidence == 0.0):
            continue
        for j in range(i + 1, objlen):
            if (IntersectionOverUnion(objects[i], objects[j]) >= 0.4):
                if objects[i].confidence < objects[j].confidence:
                    objects[i], objects[j] = objects[j], objects[i]
                objects[j].confidence = 0.0

    results = list()
    # Drawing boxes
    for obj in objects:
        if obj.confidence < 0.1:
            continue
        box = list()

        box_x = (int)(max(obj.xmin, 0))
        box_y = (int)(max(obj.ymin, 0))

        box_w = (int)(obj.xmax- obj.xmin)
        box_h = (int)(obj.ymax- obj.ymin)
        confidence = (float)(obj.confidence)
        obj_id = (int)(obj.class_id)
        track_id = 0

        box.append(box_x)
        box.append(box_y)
        box.append(box_w)
        box.append(box_h)

        box.append(confidence)
        box.append(obj_id)
        box.append(track_id)
        results.append(box)

        # label = obj.class_id
        confidence = obj.confidence
        # label_text = LABELS[label]
        # box_color, box_thickness = (0, 0, 255), 2
        # cv2.rectangle(image, (obj.xmin, obj.ymin), (obj.xmax, obj.ymax), box_color, box_thickness, 8, 0)

        # cv2.putText(image, label_text, (int((obj.xmin+obj.xmax)/2-20), obj.ymin+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 9, cv2.LINE_AA)
        # cv2.putText(image, label_text, (int((obj.xmin+obj.xmax)/2-20), obj.ymin+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 7, cv2.LINE_AA)
        # cv2.putText(image, label_text, (int((obj.xmin+obj.xmax)/2-20), obj.ymin+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 5, cv2.LINE_AA)
        # cv2.putText(image, label_text, (int((obj.xmin+obj.xmax)/2-20), obj.ymin+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        # cv2.putText(image, label_text, (int((obj.xmin+obj.xmax)/2-20), obj.ymin+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        w_ratio = (obj.xmax - obj.xmin) / image.shape[1] * 100
        h_ratio = (obj.ymax - obj.ymin) / image.shape[0] * 100

    return results
    # 100%肯定此result已包含判斷出來的動作類別的資訊了

def decode_fourcc(v):
    v = int(v)
    return "".join([chr((v >> 8*i) & 0xFF) for i in range(4)])

class PYBoxResult():
    def __init__(self):
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0
        self.prob = 0
        self.obj_id = 0
        self.track_id = 0


def thermalDetect_py(results):

    boxNum = len(results)
    PYBoxResultList = []
    for i, _ in enumerate(results):
        PYBoxResultList.append(PYBoxResult())

    for idx, member in enumerate(results):
        PYBoxResultList[idx].x = (int)(member[0]) #x
        PYBoxResultList[idx].y = (int)(member[1]) #y
        PYBoxResultList[idx].w = (int)(member[2]) #w
        PYBoxResultList[idx].h = (int)(member[3]) #h
        PYBoxResultList[idx].prob = (float)(member[4]) #final_object_score
        PYBoxResultList[idx].obj_id = (int)(member[5]) #class_w_highest_score @@@@@@@REMENBER to -1!!!! id 0 is background,
        PYBoxResultList[idx].track_id = (int)(member[6])

    return PYBoxResultList



def run_inference(input_blob, exec_net, image, DETECTION_THRESHOLD_quality_decides):

    image_to_classify = preprocess_image(image)
    # NCS2 inference!! and get the result
    outputs = exec_net.infer(inputs={input_blob: image_to_classify}) #a dict {inputs:array(1, 3, 416, 416)}

    results = post_proceesing(outputs, image, DETECTION_THRESHOLD_quality_decides) #do not use image_to_classify as input!

    print("shape:",np.array(outputs["detector/yolo-v3-tiny/Conv_12/BiasAdd/YoloRegion"]).shape)
    '''
    print("-----------------------------------------------------------------------------")
    print("shpae:",image_to_classify.shape)
    print("image_to_classify:",image_to_classify)
    print("=============================================================================")
    #print("shape:",np.array(outputs.values()))
    print("outputs:",outputs)
    print("=============================================================================")
    print("result:",results)
    print("-----------------------------------------------------------------------------")
    '''
    return results
