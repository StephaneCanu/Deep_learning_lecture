# Deep learning: an historical perspective
## Description
This repository contain the sildes of the introduction to deep learning 3h lecture given at the ECAS-ENBIS 1-Day Summer School, on the 6 of september, 2018
([00_Main_Deep_2018.pdf](https://github.com/StephaneCanu/Deep_learning_lecture/blob/master/00_Main_Deep_2018.pdf))

It comes together with practical exercices on deep learning with the solution in python based on keras

## Requirements

Keras should be available on your python environment. 
You can also install this library in the local environment using pip 
(`pip3 install keras`)

## Deep learning practical session
1. TP_Deep_1_MNIST.py (based on MNIST)
2. TP_Deep_2_webcam.py (require a web cam, and opencv-python `pip install opencv-python`)
3. TP_Deep_3_fine_tuning.py
that works with the directories contained in these zip files
   - train_cheese.zip
   - test_cheese.zip

to make it run you may:
dowload the TP_Deep_3_fine_tuning.py file to some directory and move to this directory with python (e.g. `cd ../Deep_learning_lecture`)

```
class MonArg(object):
   def __init__(self, train, val):
        self.train_dir = train
        self.val_dir = val
        self.nb_epoch = NB_EPOCHS
        self.batch_size = BAT_SIZE
        self.output_model_file = "inceptionv3-ft.model"
        self.plot = "store_true"
```
```
IM_WIDTH, IM_HEIGHT = 299, 299 
NB_EPOCHS = 25
BAT_SIZE = 32
FC_SIZE = 1024
NB_IV3_LAYERS_TO_FREEZE = 172
```
        
```
import TP_Deep_3_fine_tuning
TP_Deep_3_fine_tuning.train(MonArg("train_cheese","test_cheese"))
```
## To go further

- [A History of Deep Learning, May 30, 2018 by Andrew Fogg](https://www.import.io/post/history-of-deep-learning/)
- [A 'Brief' History of Neural Nets and Deep Learning, December 24, 2015 by Andrey Kurenkov](http://www.andreykurenkov.com/writing/ai/a-brief-history-of-neural-nets-and-deep-learning/)
- [Timeline of machine learning on wikipedia](https://en.wikipedia.org/wiki/Timeline_of_machine_learning)

