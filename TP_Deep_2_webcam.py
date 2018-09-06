# -*- coding: utf-8 -*-
"""
Slightly adapted from an original code given by Rémi Flamary
http://remi.flamary.com/index.fr.html

pip install opencv-python
pip3 install keras

This is a temporary script file.
cd Desktop/Canu/Enseign/X_Data_boot_camp/Lectures_MAP541/Lecture3_DeepLearning/

"""

import numpy as np

"""
https://www.tensorflow.org/api_docs/python/tf/keras/applications
"""


import keras
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input, decode_predictions

# https://github.com/titu1994/Keras-NASNet
#from  keras.applications.nasnet import NASNet, preprocess_input, decode_predictions
from  keras.applications.nasnet import preprocess_input, decode_predictions


# next will be
# PNASNet-5_Large_331

#%%
#model=keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)


#model=InceptionResNetV2(
#    include_top=True,
#    weights='imagenet',
#    input_tensor=None,
#    input_shape=None,
#    pooling=None,
#    classes=1000
#)



#%%
# https://github.com/titu1994/Keras-NASNet

from keras.applications.nasnet import NASNetLarge, NASNetMobile

model = NASNetLarge(input_shape=(331, 331, 3))
#model.summary()



# NASNET_LARGE_WEIGHT_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.8/NASNet-large.h5'

#%%
import pylab as pl
import cv2

cap = cv2.VideoCapture(0)

import PIL.Image

alpha=0.95
pred=0
idscreen=0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

#    img=np.array(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),(224,224)))
    img=np.array(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),(331,331)))
    
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    pred = alpha*pred+(1-alpha)*model.predict(x)
    
    txt=decode_predictions(pred,10)
    #print txt
    for i,p in enumerate(txt[0]):
        cv2.putText(frame,"{:.3f}: {}".format(float(p[2]),p[1]),(0,25*i+30),cv2.FONT_HERSHEY_PLAIN,2,[1,1,1])

    # Display the resulting frame
    cv2.imshow('frame',frame)
    key=cv2.waitKey(1)
    if (key & 0xFF) in [ ord(' '),ord('q')]:
        break
    if (key & 0xFF) in [ ord('s')]:
        cv2.imwrite("screen_{}.png".format(idscreen),frame)
        idscreen+=1


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
