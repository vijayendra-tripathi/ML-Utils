
"""
LICENSE:
========

MIT License

Copyright (c) 2018 Vijayendra Tripathi

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


DESCRIPTION:
============
A Callback subclass created to write output to a file after every epoch finishes.
This is handy if we loose internet connection to remote server. The server 
will write output to a file and then we can recover all logs to see what 
happened in the background.

REFERENCES:
===========
References: https://keras.io/callbacks/
You can also use keras CSVLogger for this purpose, but it do not include
time information for every epoch. 

NOTE:
=====
It is recommended to run notebook using following command to keep notebook
running even if SSH connection to remote server is closed.

nohup jupyter notebook --ip=0.0.0.0 --no-browser 

This video explains it - https://www.youtube.com/watch?v=q4PrYQOShnE

USAGE EXAMPLE:
==============
# 1. Import logger
from ML_Utils import RemoteTrainingLogger

# 1. Initialize
logger = RemoteTrainingLogger('training_logs.txt') 

# 2. Include logger in training
model.fit(x_data, y_data, batch_size=500, epochs=30, verbose=1, callbacks=[logger])

"""

from keras.callbacks import Callback
import time

class RemoteTrainingLogger(Callback):
    def __init__(self, file_name):
        self.all_logs = []
        self.training_start_time = time.time()
        self.begin_epoch_time = 0
        self.file_name = file_name

    def on_epoch_begin(self, epoch, logs={}):
        self.begin_epoch_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        end_epoch_time = time.time() - self.begin_epoch_time
        log = "Epoch: " + str(epoch + 1) + " Time: " + str(int(end_epoch_time)) + "s Loss: " + str(round(logs.get('loss'), 4)) + "\n"
        self.all_logs.append(log)

    def on_train_end(self, logs):
        training_end_time = time.time()
        total_training_time = training_end_time - self.training_start_time
        log = "Total Training Time: " + str(int(total_training_time)) + "s" + "\n"
        self.all_logs.append(log)
        self.save()
        # print('--training ended--')

    def on_train_begin(self, logs):
        self.all_logs.clear()  # Clear all existing logs
        self.training_start_time = time.time()

    def save(self):
        with open(self.file_name, "w") as f:
            for txt in self.all_logs:
                f.write(txt)
        self.all_logs.clear() # Clear all existing logs
