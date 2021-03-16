# EfficientNet B4 Classifier

Code example from: Image classification via fine-tuning with EfficientNet
https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/

## Load Dataset to TensorFlow
See `Fruit360Dataset.py` for this step.

We use the Fruits 360 dataset on GitHub:\
https://github.com/srbhr/Fruits_360/blob/master/Fruits_Detection.ipynb

First download the dataset, I put it in the `../Dataset` folder. 

Because the dataset that we use is not a pre-built dataset for TensorFlow, we need to load it to the pipeline ourselves.
TensorFlow considers this as a "set of files".

More instructions on how to load a folder as dataset source from the documentation:
https://www.tensorflow.org/guide/data#consuming_sets_of_files

The dataset is split into 80% train and 20% test. The method `Fruit360Dataset.fruit360_dataset` returns
train and test datasets after split, and labels of the fruit.

## EfficientNet Classification
Some parameters used by EfficientNet B4 parameters for training"

**Image size: 380  (commended by EfficientNet B4)**

**Batch size (load dataset): 64**

Image augmentation:
**Random rotation factor: 0.15**

Image augmentation:
**Random translation height factor: 0.1**
**Random translation height width factor: 0.1**

Image augmentation:
**Random contrast factor: 0.1**

### EfficientNetB4 function
include_top=True: Whether to include the fully-connected layer at the top of the network. Defaults to True.
weights=None: One of None (random initialization), 'imagenet' (pre-training on ImageNet), or the path to the weights
 file to be loaded. Defaults to 'imagenet'.\
classes=131

### Compilation Options
optimizer: ADAM optimizer\
loss function: categorical cross entropy\
evaluation metrics: accuracy

Number of epochs: 10   (10 is the minimum, I used it because it took sooooooo looooong)
Top dropout rate: 0.2


# Result (10 epochs)
loss: 4.3
accuracy: 0.042
val_loss: 4.4
val_accuracy: 0.040
