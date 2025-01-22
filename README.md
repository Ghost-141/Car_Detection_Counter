# ![Car Detection & Counter Logo](https://github.com/Ghost-141/Car_Detection_Counter/blob/main/cover.png)[Car Counting and Detection Using OpenCV in Real Time]
This repository represents a Deep Learning-based system designed to detect and count vehicles in a video stream. Utilizing a YOLOv8 model, the system classifies vehicles into different categories such as cars, buses, trucks, and motorbikes. 

## Table of Contents
- [Introduction](#introduction)
- [Usage](#usage)
- [Features](#features)
- [Library Installation](#library-installation)
- [Conclusion](#conclusion)

## Introduction
This projects utilizes the power of YOLOV8 and OpenCV model  to Classify between different vehicles across different classes and counting them in real time. This project is ideal for real-time traffic monitoring and analysis, providing insights into vehicle counting and density.

## Usage
- Download the folder and install all the libraries mentioned in library section
- Ensure that the [`yolov8s.pt`](yolov8s.pt) or any of the yolo model file is in the project directory
- Run the [`main.py`](main.py) file to start counting and detection from any video footage.
   
## Features
- Classify vehicles with their class names and confidence score in real time from video footage
- Count the number of cars passing through each lane precisely
- Can be utilized further with multiple lanes in roads

## Library Installation:  
- `OpenCV`
- `math`
- `numpy`
- `cvzone`
- `sort`
- `Ultralytics` 

These are mainly used to build this project. But there are other dependencies which will be installed automatically while installing them from the yml file. Make sure to install cuda(11.8/12.6) for GPU support if you have a dedicated gpu in your system. You can install the above mentioned libraries with specific version from [`environment.yml`](environment.yml).

For Conda installation:
- make sure to run the conda command prompt in `Administrator` mode 
- To create a new environment with all the required libraries
\`\`\`
conda env create -n my_new_env -f environment.yml
\`\`\`
- To install required libraries in existing conda environment(`my_new_env`) 
\`\`\`
conda env update -n my_new_env -f environment.yml
\`\`\`
## Conclusion
- Having any issue or question feel free to reach out
- Please give it a star if you find it useful


