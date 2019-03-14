#! /usr/bin/env python

import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from preprocessing import parse_annotation
from utils import draw_boxes
from frontend import YOLO
import json
import pandas as pd
#create a data frame - dictionary is used here where keys get converted to column names and values to row values.


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


def _main_():
    config_path  = 'config.json'
    weights_path = 'model.h5'
    image_path   = 'image.mp4'

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    ###############################
    #   Make the model 
    ###############################

    yolo = YOLO(backend             = config['model']['backend'],
                input_size          = config['model']['input_size'], 
                labels              = config['model']['labels'], 
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'])

    time_now = 0
    data_head = pd.DataFrame({'Time': [0],'Head_count':[0]})
    
    ###############################
    #   Load trained weights
    ###############################    

    yolo.load_weights(weights_path)

    ###############################
    #   Predict bounding boxes 
    ###############################

    if image_path[-4:] == '.mp4':
        video_out = image_path[:-4] + '_detected' + image_path[-4:]
        video_reader = cv2.VideoCapture(image_path)

        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

        video_writer = cv2.VideoWriter(video_out,
                               cv2.VideoWriter_fourcc(*'MPEG'), 
                               50.0, 
                               (frame_w, frame_h))

        for i in tqdm(range(nb_frames)):
            _, image = video_reader.read()
            time_now +=1
            boxes = yolo.predict(image)
            data_head = data_head.append({'Time': str(time_now//60)+'/'+str(time_now%60),'Head_count': len(boxes)}, ignore_index=True)
            image = draw_boxes(image, boxes, config['model']['labels'])

            video_writer.write(np.uint8(image))
            if(time_now==62):
                break

        video_reader.release()
        video_writer.release()
        data_head.to_csv('head_count.csv',index=False)

    else:
        image = cv2.imread(image_path)
        boxes = yolo.predict(image)
        image = draw_boxes(image, boxes, config['model']['labels'])
        data_head = data_head.append({'Time': '0/1','Head_count':len(boxes)},ignore_index=True)

        print(len(boxes), 'boxes are found')
        data_head.to_csv('head_count.csv',index=False)
        cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], image)
       

if __name__ == '__main__':
    _main_()
