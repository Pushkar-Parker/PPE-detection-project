# PPE Detection Model (YOLO)
This repository contains a YOLO-based Personal Protective Equipment (PPE) detection model trained to classify and detect helmets and reflective vests in images and videos. The model is designed to enhance workplace safety by identifying whether individuals are wearing the required PPE.

**Dataset & Classes**

The dataset used for training: https://universe.roboflow.com/work-safe-project/safety-vest---v4
The model was trained on a custom dataset with the following four classes:

* Helmet – Person wearing a safety helmet
* Non-Helmet – Person without a helmet
* Reflective – Person wearing a reflective vest
* Non-Reflective – Person without a reflective vest

**Model Details**
* Model Architecture: YOLO (You Only Look Once)
* Framework: Ultralytics YOLO (PyTorch)
* Image Size: 640x640
* Training Epochs: 50
