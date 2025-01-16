# Tkinter YOLO Video Monitor

![alt text](https://github.com/blewett/monitor-tkinter-yolo/blob/main/jeopardy.png?raw=true)

This system is built on the Open Computer Vision module (cv2), the YOLO (you only look once) image detection system, and the  tkinter GUI package. Install these, after installing python3, with the following:

1. pip3 install opencv-python
2. sudo apt-get install python3-tk
3. pip3 install --upgrade Pillow
4. Download the YOLO data from many of the online sources:

https://medium.com/@mikolaj.buchwald/yolo-and-coco-object-recognition-basics-in-python-65d06f42a6f8

https://data-flair.s3.ap-south-1.amazonaws.com/machine-learning-projects/pedestrian-detection-project-code.zip

The following is a sample invocation of the system:

python3 monitor.py

There is a wide interest in using video cameras for monitoring.  The YOLO system has 80 different objects that it can detect.   The tkinter interface on this system allows one to search for those objects by selecting the object from the pull down menu.

![alt text](https://github.com/blewett/monitor-tkinter-yolo/blob/main/tkinter-yolo-opencv.png?raw=true)

There are two windows: one displaying the raw video stream and one showing matches that have been detected.  Video and/or jpeg files may be generated through the interface.  Video recording starts from the match screen when an object is detected.

I hope this helps.  You are on your own – but you already knew that.

Doug Blewett

doug.blewett@gmail.com
