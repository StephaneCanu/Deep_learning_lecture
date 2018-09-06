# Deep learning lecture
Deep learning lecture

00_Main_Deep_2018.pdf
is the introduction to deep learning 3h lecture given at the ECAS-ENBIS 1-Day Summer School, on the 6 of september, 2018

It comes together with 3 practical exercices in python based on keras
(`pip3 install keras`)

1. TP_Deep_1_MNIST.py (based on MNIST)
2. TP_Deep_2_webcam.py (require a web cam, and opencv-python `pip install opencv-python`)
3. TP_Deep_3_fine_tuning.py
that works with the directories contained in these zip files
   - train_cheese.zip
   - test_cheese.zip

to make it run you may:
dowload the TP_Deep_3_fine_tuning.py file to some directory adn move to this directory with python (e.g. `cd ../Deep_learning_lecture`)

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
