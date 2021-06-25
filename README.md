# Object_detection_in_satellite_images
# FASTER-RCNN with DOTA
We provide the config files, TFRecord files and label_map file used in training DOTA with FASTER-RCNN, and the trained models have been uploaded to google Drive.
Notice that there are in google colab step by step process from mounting the drive until evaluation of the result.

Installation
Tensorflow:
    !pip install tensorflow==1.15.2
Object Detection API
Follow the instructions in Installation. Note the version of Protobuf.
Development kit
You can easily install it following the instructions in readme.
Preparing inputs
Select one class, clean the selected data  and then converted to tfrecord 

# From tensorflow/models/object_detection/
python create_dota_tf_record.py \
    --data_dir=/your/path/to/dota/train \
    --indexfile=train.txt \
    --output_name=dota_train.record \
    --label_map_path=data/dota_label_map.pbtxt \
The subdirectory of "dataset" is in the structure of

dataset
    ├── images
    └── labelTxt
    └── indexfile
Here the indexfile contains the full path of all images to convert, such as train.txt or test.txt. Its format is shown below. This file is an input file for the create_dota_tf_record.py script 

/your/path/to/dota/train/images/P0035__1__0___0.png
/your/path/to/dota/train/images/P0035__1__0___595.png
...
And the output path of tf_record is also under "dataset"

Labelmap
In this case the labelmap containe only one class but it can change depend of the nomber of class

PIPLINE CONFIG
in this file you should change:
- the number of class to your need
- add the path of object detection model.ckpt
- add the paths of train.record and val.record
- add the path to the labelmap


Training

# From tensorflow/models/object_detection/
!python train.py \
    --logtostderr \
    --pipeline_config_path=${PATH_TO_YOUR_PIPELINE_CONFIG} \
    --train_dir=${PATH_TO_TRAIN_DIR}
The pipline config file for DOTA data set can be found in the Google drive.

Here we train FASTER-RCNN with image size of 1024×1024, YOLOv5 with image size of 608×608. Please refer to DOTA_devkit/ImgSplit.py to split the picture and label. The trained models can be downloaded here:

Google Drive: [FASTER-RCNN](https://drive.google.com/drive/u/1/folders/0AKXPJlD12Pn_Uk9PVA), [YOLOv5](https://drive.google.com/drive/u/1/folders/0ACEYqbc5R1cSUk9PVA)
Evaluation
You can use the pre-trained models to test images. 


#  YOLOv5
For this one we followed the set from the Ultralytics YOLOv5 Repo.
and in the google colab you will find step by step process from mounting the drive until evaluation of the result.
you can find all the scripts in the script folder
we converted to DOTA dataset to the yolo. 

before starting the training the structure of the data subdirectory should be like that :
data
    ├── images
        └── train
        └── val
    ├── labels
        └── train
        └── val

Then you have to create dota.yaml file it structure is as follow :
 - train: ‘training set directory path’
 - val: ‘validation set directory path’
 - nc: ‘number of classes’
 - names: ‘name of objects’
and you have to change in the dowloaded repository yolov5/model/: the number of class in the yaml files
Download the weight for YOLOv5
!sh -x weights/download_weights.sh
For training and detection the are described in the Google colab

#  Flask
## Flask with Faster RCNN

Download Tensorflow API from Github Repository
Setting up a virtual environment
  - To set up the virtual environment:
conda create -n obj_detection

  - To activate it the above created virtual environment:
conda activate obj_detection

Installing dependencies
The next step is to install all the dependencies needed for this API to work on your local PC. Type this command after activating your virtual environment.

You should have python 3.7 to be able to install tensorflow ==1.15.2
pip install tensorflow==1.15.2
If you have a GPU in your PC, use this instead. You will have a better performance

pip install tensorflow-gpu
Next, use this command to install the rest of dependencies

pip install pillow Cython lxml jupyter matplotlib contextlib2 tf_slim
download Protocol Buffers (Protobuf) then execute these commands
o protoc object_detection/protos/*.proto --python_out=.
o python setup.py build
o python setup.py install
For installing pycocotools there are two  ways to so :
o pip install pycocotools (or) pip install git+https://github.com/philferriere/cocoa...^&subdirectory=PythonAPI

After installing the object detction api 
you should intall Flask 
in the following folder: models/research/ 
add the following folders:
* templates
*static
*inference_graph
*training

to run the app : python detction.py

## Flask with YOLOv5

Download Ultralytics YOLOv5 from Github Repository
Setting up a virtual environment
  - To set up the virtual environment:
conda create -n obj_detection

  - To activate it the above created virtual environment:
conda activate obj_detection

Installing dependencies
cd yolov5
pip install -qr requirements.txt 
in the current folder add the following folders:
* templates
* static

to run the app : python detect.py



