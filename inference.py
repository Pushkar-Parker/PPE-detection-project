# Importing modules
from ultralytics import YOLO
import random
import numpy as np
import os
import torch
from pathlib import Path
import cv2 as cv

# Setting up random seed
random.seed(101)

# Alloting each class a unique color
def classes_color(classes: dict):
    color_dict = {}

    for i in range(len(classes)):
        random.seed(i)
        color_dict[i] = random.sample(range(256), 3)
        
    return color_dict

# Inference function
def object_detection(model_path: str, # model path 
                     test_images_path: str, # path for the test
                     pred_save_path: str, # path to save predictions
                     ):

    # Device agnostic code
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # model function
    def get_model(model_path: str):
        model = YOLO(model_path).to(device)
        return model

    # executing the model function
    model = get_model(model_path)

    # Accessing the test images from the path
    test_images_path = Path(test_images_path)
    test_images = list(test_images_path.glob('*.jpg'))
    
    # creating save directory
    if not os.path.exists(pred_save_path):
        os.mkdir(pred_save_path)

    # Iterating through the test images    
    for i, image in enumerate(test_images):
        
        # Running inference on the image
        results = model(image)
        
        img = results[0].orig_img # the original image
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(np.int32) # bounding box co-ordinates
        labels = results[0].boxes.xywhn.cpu().numpy() # co-ordinates in xywhn format
        classes = results[0].boxes.cls.cpu().numpy().astype(np.int32) # int representing classes
        classes_name = results[0].names # classes names
        labels_color = classes_color(classes_name) # unique color for each class
        
        image_filename = f'{test_images[i].stem}.jpg' # setting up image file name
        
        for i, (bbox, cls) in enumerate(zip(boxes, classes)): # iterating over bounding boxes and classes
    
            label = classes_name[cls] # label name 
            label_margin = 3 # setting up label background margin
            label_size = cv.getTextSize(label, 
                                        fontFace=cv.FONT_HERSHEY_SIMPLEX, 
                                        fontScale=0.45, 
                                        thickness=1) # getting the label text size for the given parameters
            
            # Defining the margins
            label_w, label_h = label_size[0] 
            label_w += 2*label_margin
            label_h += 2*label_margin

            # Drawing bounding box on the image
            cv.rectangle(img, 
                         (bbox[0], bbox[1]), 
                         (bbox[2], bbox[3]), 
                         color= labels_color[cls], 
                         thickness=1)
            
            # Drawing the label background
            cv.rectangle(img, 
                         (bbox[0], bbox[1]), 
                         (bbox[0]+label_w, bbox[1]-label_h), 
                         color= labels_color[cls], 
                         thickness=-1)

            # Putting the label text on the image
            cv.putText(img, 
                       label, 
                       (bbox[0]+label_margin, bbox[1]-label_margin), 
                       fontFace=cv.FONT_HERSHEY_SIMPLEX, 
                       color=(255, 255, 255), 
                       fontScale= 0.45, 
                       thickness=1)

        # Saving the image
        cv.imwrite(os.path.join(pred_save_path, image_filename), img)
        print(f'{image_filename} saved in {pred_save_path}')

    print('\nDone')

# Executing the function
object_detection(model_path="model_path", 
                 test_images_path="test_images_path", 
                 pred_save_path="pred_save_path")
