"""
monitor.py: Original work Copyright (C) 2024 by Blewett

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This is a simple camera monitor interface.

An elaborate recorgnizer is from "data-flair":
 from https://data-flair.training/blogs/pedestrian-detection-python-opencv/

 https://arxiv.org/pdf/1506.02640
 https://medium.com/@mikolaj.buchwald/yolo-and-coco-object-recognition-basics-in-python-65d06f42a6f8

A smpler recognizer is from the cv2 supported haarcasade recognizer:
haarcascade_frontalface_default.xml

The system can write jpegs and/or video files and do face detection.

Load cv2:
 pip3 install opencv-python

Load tkinter:
 apt-get install python3-tk

Load Pillow:
 pip3 install --upgrade Pillow

and other words like that.
"""
import tkinter
import tkinter as tk
from tkinter import ttk

import cv2
import PIL.Image, PIL.ImageTk
import numpy as np
import time
import datetime
import threading
from collections import deque

# global queues for communicating frame
global stream_queue
global matching_queue

def date_time_string():
    t = datetime.datetime.now().strftime("%f")[:-5]
    f = datetime.datetime.now().strftime("%H-%M-%S." + t + "  %d/%m/%Y")
    return f

def label_frame(frame, width, height):
    # font type and dimensions
    font = cv2.FONT_HERSHEY_SIMPLEX 
    font_scale = 0.5
    font_thickness = 3
    (label_width, label_height), label_baseline = cv2.getTextSize("M", font, font_scale, font_thickness)

    x = label_width
    y = height - (label_height + label_baseline)

    ds = date_time_string()

    # make it visible: red and blue and two pixels wide
    cv2.putText(frame, ds, (x,y), font, font_scale, (255, 0, 255), 2)


# FIFOqueue for communicating with thread safe queues
class FIFOQueue:
    def __init__(self):
        self.queue = deque()

    def enqueue(self, item):
        self.queue.append(item)

    def dequeue(self):
        if self.is_empty():
            return None

        item = self.queue.popleft()
        return item

    def printqueue(self):
        print(self.queue)

    def peek(self):
        if self.is_empty():
            return None
        return self.queue[0]

    def is_empty(self):
        return len(self.queue) == 0

    def size(self):
        return len(self.queue)
# end FIFOqueue for communicating with thread safe queues

# VideoStream for reading and processing video stream - including files
class VideoStream:
    global stream_queue
    global matching_queue

    def __init__(self, video_source=0, width=None, height=None, fps=None):
    
        self.video_source = video_source
        self.width = width
        self.height = height
        self.fps = fps

        # Open the video source
        self.vid = cv2.VideoCapture(video_source)

        # check if camera opened successfully
        if not self.vid.isOpened():
            print("cv2.VideoCapture() cannot open the camera: " +
                  str(video_source))
            exit()

        # Get video source width and height
        # convert float to int
        if not self.width:
            self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        if not self.height:
            self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not self.fps:
            self.fps = int(self.vid.get(cv2.CAP_PROP_FPS))

        # slow things down as we are processing the cv2 frames
        self.fps = int(self.fps / 2)

        #
        # load the data-flair-training detection system
        #
        self.NMS_THRESHOLD=0.3
        self.MIN_CONFIDENCE=0.2

        weights_path = "yolov4-tiny.weights"
        config_path = "yolov4-tiny.cfg"

        self.model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        '''
        self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        '''

        self.layer_name = self.model.getLayerNames()
        self.layer_name = [self.layer_name[i - 1] for i in self.model.getUnconnectedOutLayers()]

        # set default - overriden later
        labelsPath = "coco.names"
        self.LABELS = open(labelsPath).read().strip().split("\n")
        self.object_index = self.LABELS.index("person")

        #
        # end flair-training detection system
        #

        #
        # load the haarcascade detection system
        #
        """
        cascPath = "haarcascade_frontalface_default.xml"
        self.faceCascade = cv2.CascadeClassifier(cascPath)
        """

        #
        # end haarcascade detection system
        #

        # default frame values at start        
        self.ret = False
        self.frame = None
        
        self.convert_color = cv2.COLOR_BGR2RGB
        self.convert_pillow = True

        # add frames to both fifos self.match_double_count for adding trailers
        #  when recording video
        self.match_double_count = 0
        self.matching_video = False
        self.match_added_seconds = 3
        self.match_added_frames = int(self.fps * self.match_added_seconds)

        # start thread
        self.running = True
        self.thread = threading.Thread(target=self.process)
        self.thread.start()

    def data_flair_detection(self, image, model, layer_name, object_index):
        (H, W) = image.shape[:2]
        #
        # converted results return to rectangles
        #
        # results = []
        rects = []
        rectl = []

        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        model.setInput(blob)
        layerOutputs = model.forward(layer_name)

        boxes = []
        centroids = []
        confidences = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if classID == object_index and confidence > self.MIN_CONFIDENCE:

                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    centroids.append((centerX, centerY))
                    confidences.append(float(confidence))
                    # apply non-maxima suppression to suppress weak, overlapping
                    # bounding boxes
                    idzs = cv2.dnn.NMSBoxes(boxes, confidences,
                                            self.MIN_CONFIDENCE,
                                            self.NMS_THRESHOLD)
                    # ensure at least one detection exists
                    if len(idzs) > 0:
                        # loop over the indexes we are keeping
                        for i in idzs.flatten():
                            # extract the bounding box coordinates
                            (x, y) = (boxes[i][0], boxes[i][1])
                            (w, h) = (boxes[i][2], boxes[i][3])
                            # update our results list to consist of the person
                            # prediction probability, bounding box coordinates,
                            # and the centroid
                            # res = (confidences[i], (x, y, x + w, y + h), centroids[i])
                            # results.append(res)
                            rects.append([x, y, w, h])
                            # return the list of results
                    #return results
        return rects


    def process(self):
        global stream_queue
        global matching_queue

        while self.running:
            ret, frame = self.vid.read()
            
            if ret:
                frame = cv2.resize(frame, (self.width, self.height))

                """
                #
                # haarcascade processing
                #

                # Convert the frame to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect rects in the grayscale frame
                rects = self.faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                )
                #
                # end haarcascade processing
                #
                """

                #
                # start data_flair processing
                #
                rects = self.data_flair_detection(frame, self.model,
                                                  self.layer_name,
                                                  self.object_index)
                #
                # end data_flair processing
                #

                # Draw a rectangle around the detected object
                if rects and len(rects) > 0:
                    for (x, y, w, h) in rects:
                        cv2.rectangle(frame, (x, y), (x + w, y + h),
                                      (0, 255, 0), 2)

                label_frame(frame, self.width, self.height)

                if self.convert_pillow:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = PIL.Image.fromarray(frame)
                    stream_queue.enqueue(frame)

                if rects and len(rects) > 0:
                    matching_queue.enqueue(frame)
                    if self.matching_video:
                        self.match_double_count = self.match_added_frames
                elif self.matching_video == True and self.match_double_count > 0:
                    self.match_double_count = self.match_double_count - 1
                    matching_queue.enqueue(frame)

            else:
                self.message_label.config(text = "[ video stream end ]")
                self.running = False
                break

            # assign new frame
            self.ret = ret
            self.frame = frame

            # sleep for next frame
            time.sleep(1/self.fps)

    def get_frame(self):
        return self.ret, self.frame
# end VideoStream for reading and processing video stream - including files


# tkMonitorPanel tkInter based thread for displaying and controling video
class tkMonitorPanel(tkinter.Frame):
    def __init__(self, window, video_stream, message_label, label="", width=None, height=None, frame_queue=None, video_fps=16):

        super().__init__(window)
        
        self.window = window
        self.width = width
        self.height = height
        self.fps = video_fps
        self.message_label = message_label

        self.label_text = label
        self.frame_queue = frame_queue
        self.video_stream = video_stream

        self.label = tkinter.Label(self, text=label)
        self.label.pack()
        
        self.canvas = tkinter.Canvas(self, width=400, height=300)
        self.canvas.pack()

        # write jpgs of matches or streams
        self.write_jpgs = tkinter.IntVar(value=0)
        self.checkbox_write_jpgs = tkinter.Checkbutton(self, text="write jpgs",
                                                       variable=self.write_jpgs)
        self.checkbox_write_jpgs.pack(anchor='center', side='left')

        self.write_video = tkinter.IntVar(value=0)
        self.checkbox_write_video = tkinter.Checkbutton(self,
                                                        text="write a video",
                                                        command=self.write_video_f,
                                                        variable=self.write_video)
        self.recording_filename = None
        self.recording_writer = None

        self.checkbox_write_video.pack(anchor='center', side='left')
        self.write_video_count = 0
        self.recording = False

        # Checkbutton that lets the user take a snapshot
        self.btn_snapshot = tkinter.Button(self, text="take a snapshot",
                                           command=self.snapshot_f)
        self.btn_snapshot.pack(anchor='center', side='left')
        
        # After it is called once, the update method will be automatically
        #  called every delay milliseconds
        #  assume at best 12 frames per second - your milage may differ
        self.delay = int(1000/16)

        self.image = None
        self.running = True
        self.update_frame()

	# read default image
        if self.image == None:
            self.photo = PIL.ImageTk.PhotoImage(file="sheep-dog-working-from-home.jpg")
            self.canvas.create_image(0, 0, image=self.photo, anchor='nw')

        self.thread = threading.Thread(target=self.update_frame)
        self.thread.start()


    def snapshot_f(self):
        # save the last image that was displayed in tkMonitorPanel
        if self.image:
            self.image.save(time.strftime("snap-" + self.label_text +
                                          "-%d-%m-%Y-%H-%M-%S.jpg"))
            
    def write_video_f(self):
        if self.write_video.get() == 1:
            if self.recording == True:
                self.message_label.config(text = "[ video recording already running ]")
                return

            self.start_recording()
        else:
            self.stop_recording()


    def start_recording(self, filename=None):

        if self.recording:
            self.message_label.config(text = "[  already recording: " + self.recording_filename + " ]")
        else:
            # These bits are based on the work of Bartłomiej “furas” Burek
            # (link provied below).
            #
            # https://stackoverflow.com/questions/65876044
            #  /how-display-multi-videos-with-threading-using-tkinter-in-python
            #
            # fine work by Furas from Poland - Bartłomiej “furas” Burek
            #   https://blog.codersrank.io/polish-developers-you-should-follow
            #   /#h-bart-omiej-furas-burek
            #
            # https://github.com/furas?tab=repositories
            #

            # VideoWriter constructors
            #.mp4 = codec id 2
            if filename:
                self.recording_filename = filename
            else:
                self.recording_filename = time.strftime(self.label_text +
                                                        "-%d-%m-%Y-%H-%M-%S.avi")

            #fourcc = cv2.VideoWriter_fourcc(*'I420') # .avi
            #fourcc = cv2.VideoWriter_fourcc(*'MP4V') # .avi
            fourcc = cv2.VideoWriter_fourcc(*'MP42') # .avi
            #fourcc = cv2.VideoWriter_fourcc(*'AVC1') # error libx264
            #fourcc = cv2.VideoWriter_fourcc(*'H264') # error libx264
            #fourcc = cv2.VideoWriter_fourcc(*'WRAW') # error --- no information ---
            #fourcc = cv2.VideoWriter_fourcc(*'MPEG') # .avi 30fps
            #fourcc = cv2.VideoWriter_fourcc(*'MJPG') # .avi
            #fourcc = cv2.VideoWriter_fourcc(*'XVID') # .avi
            #fourcc = cv2.VideoWriter_fourcc(*'H265') # error 
            self.recording_writer = cv2.VideoWriter(self.recording_filename, fourcc, self.fps, (self.width, self.height))
            self.recording = True
            if self.label_text == "match":
                self.video_stream.matching_video = True

            # start match recording with one second trailer - subclass?
            #self.window.message_label.config(text = "[ start recording: " + self.recording_filename + " ]")
            self.message_label.config(text = "[ started recording: " + self.recording_filename + " ]")


    def stop_recording(self):
        if not self.recording:
            self.message_label.config(text = "[ not currently recording ]")
        else:
            self.recording = False
            self.recording_writer.release() 
            self.message_label.config(text = "[ stop recording: " + self.recording_filename + " ]")
            if self.label_text == "match":
                self.video_stream.matching_video = False


    def update_frame(self):
        # Get a frame
        if self.frame_queue.is_empty() == True:
            if self.running:
                self.window.after(self.delay, self.update_frame)
                return

        self.image = self.frame_queue.dequeue()
        self.photo = PIL.ImageTk.PhotoImage(image=self.image)
        self.canvas.create_image(0, 0, image=self.photo, anchor='nw')

        # write videos
        if self.recording:
            if self.recording_writer and self.recording_writer.isOpened():
                cv_img = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)
                self.recording_writer.write(cv_img)

        # write jpgs
        if self.write_jpgs.get() == 1:
            f = datetime.datetime.utcnow().strftime("-%d-%m-%Y-%H-%M-%S.%f")[:-4]
            self.image.save(self.label_text + f + ".jpg")

        if self.running:
            self.window.after(self.delay, self.update_frame)
# end tkMonitorPanel tkInter based thread for displaying and controling video

class App:

    def __init__(self, window, window_title):
        global stream_queue
        global matching_queue

        self.window = window
        self.window.title(window_title)
        
        stream_queue = FIFOQueue()
        matching_queue = FIFOQueue()

        self.video_source = 0
        self.video_stream = VideoStream(self.video_source, 400, 300)
        self.video_stream.matching_video = False

        # frame = tk.Frame(self.window,  width=300, height=200)
        frame = tk.Frame(self.window)
        frame.pack(expand=True, padx=10, pady=10)

        # Create the Label
        label = tk.Label(frame, text="Select the search object:")
        label.grid(row=0, column=0, padx=5, pady=5)

        # Create the Combobox
        # Load words from a file
        filename = "coco.names"
        (words_list, words_dict) = self.load_words_from_file(filename)
        selected_word_var = tk.StringVar()
        combobox = ttk.Combobox(frame, textvariable=selected_word_var)
        combobox['values'] = words_list

        # set default as person
        combobox.current(48)
        combobox.bind('<<ComboboxSelected>>', lambda event: self.on_word_selected(event, words_dict))
        combobox.grid(row=0, column=1, padx=5, pady=5)

        # Center the frame on the left of the root frame
        frame.grid_columnconfigure(0, weight=1)
        frame.grid(row=0, column=0)

        # Create the message label - on the right
        self.message_label = tk.Label(self.window, text="[ running ]")
        self.message_label.grid(row=0, column=1, padx=5, pady=5)

        self.tkMonitorPanel1 = tkMonitorPanel(self.window, self.video_stream, self.message_label, "stream", 400, 300, stream_queue, self.video_stream.fps)
        self.tkMonitorPanel1.grid(row=1, column=0)

        self.tkMonitorPanel2 = tkMonitorPanel(self.window, self.video_stream, self.message_label, "match", 400, 300, matching_queue, self.video_stream.fps)
        self.tkMonitorPanel2.grid(row=1, column=1)

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()

    def on_closing(self, event=None):
        self.message_label.config(text = "[ stopping threads ]")
        print('[App stopping threads ]')

        self.video_stream.running = False
        self.tkMonitorPanel1.running = False
        self.tkMonitorPanel2.running = False

        self.message_label.config(text="[ App exit ]")
        print('[App] exit')
        self.window.destroy()
        exit()

    def load_words_from_file(self, filename):
        # load indexed words
        try:
            with open(filename, 'r') as file:
                words = [line.strip() for line in file.readlines()]
        except FileNotFoundError:
            print(f"Index file '{filename}' not found.")
            exit(1)

        words_dict = {}
        i = 0
        for w in words:
            words_dict[w] = i
            i += 1

        words = sorted(words)

        return words, words_dict

    def on_word_selected(self, event, words_dict):
        selected_word = event.widget.get()
        selected_index = words_dict.get(selected_word, None)
        self.video_stream.object_index = self.video_stream.LABELS.index(selected_word)
        self.message_label.config(text = "[ searching for " + selected_word + " ]")


if __name__ == '__main__':     

    App(tkinter.Tk(), "Camera Monitor with Tkinter and OpenCV")
