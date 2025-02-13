# Importing modules
import torch
from ultralytics import YOLO
import os

# Device agnostic code
device = 'cuda:0' if torch.cuda.is_available else 'cpu'

# model training function
def train_model(model: str, # the model path
                data: str, # the .yaml file
                epochs: int, # the number of epochs
                img_size: int, # the image size for training
                save_path: str, # save path for training 
                experiment: str, # experiment name
                device: torch.device= device # defining the device to train on
                ):
    
    # creating save derectory
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    # defining model
    model = YOLO(model).to(device)

    # model training
    model.train(
        data= data,
        epochs= epochs,
        imgsz= img_size,
        device= device,
        project= save_path,
        name= experiment
    )

if __name__ == '__main__':
    train_model(
        model=r"D:\Safety vest - v4.v3i.yolov11\training_results\50_epochs\weights\best.pt",
        data=r"D:\Safety vest - v4.v3i.yolov11\data.yaml",
        epochs=20,
        img_size=640,
        save_path=r"D:\Safety vest - v4.v3i.yolov11\training_results",
        experiment='50_epochs_3'
    )