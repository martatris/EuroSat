# EuroSAT Land Use and Land Cover Classification


## Overview
--------
This project performs Land Use and Land Cover (LULC) classification using the EuroSAT dataset.
It uses a Convolutional Neural Network (CNN) built in TensorFlow/Keras to classify satellite images
into different land cover categories such as Forest, Industrial, Residential, etc.

## Dataset
-------
You should have the EuroSAT dataset organized as follows:

    EuroSAT/
        AnnualCrop/
        Forest/
        HerbaceousVegetation/
        Highway/
        Industrial/
        Pasture/
        PermanentCrop/
        Residential/
        River/
        SeaLake/

Each subfolder contains images belonging to that land use category.

## Main Script
------------
The main Python file is `EuroSat.py`.

Key Steps in the Code:
1. Loads the dataset using `image_dataset_from_directory`.
2. Preprocesses and batches the data.
3. Builds a CNN model.
4. Trains the model on the EuroSAT dataset.
5. Evaluates accuracy and plots results.

## Dependencies
-------------
Install these packages before running the script:

    pip install tensorflow matplotlib

Optional for better performance:
    pip install tensorflow-datasets scikit-learn protobuf==3.20.*

## Running the Project
-------------------
1. Make sure your data directory path is correctly set in the script:
       DATA_DIR = "/Users/<yourname>/Desktop/Data/EuroSAT"
2. Run the script:
       python EuroSat.py
3. The model will train for several epochs and display accuracy results and training graphs.

## Output
------
- Training and validation accuracy per epoch.
- Final validation accuracy printed in the console.
- A plot showing accuracy trends over epochs.

```
Epoch 1/10
675/675 [==============================] - 29s 41ms/step - loss: 1.4184 - accuracy: 0.4553 - val_loss: 0.8871 - val_accuracy: 0.7006
Epoch 2/10
675/675 [==============================] - 27s 40ms/step - loss: 0.9750 - accuracy: 0.6531 - val_loss: 0.7183 - val_accuracy: 0.7413
Epoch 3/10
675/675 [==============================] - 28s 41ms/step - loss: 0.8257 - accuracy: 0.7080 - val_loss: 0.6575 - val_accuracy: 0.7657
Epoch 4/10
675/675 [==============================] - 29s 43ms/step - loss: 0.7382 - accuracy: 0.7430 - val_loss: 0.6042 - val_accuracy: 0.7943
Epoch 5/10
675/675 [==============================] - 29s 43ms/step - loss: 0.6620 - accuracy: 0.7697 - val_loss: 0.5354 - val_accuracy: 0.8124
Epoch 6/10
675/675 [==============================] - 30s 44ms/step - loss: 0.5951 - accuracy: 0.7946 - val_loss: 0.4988 - val_accuracy: 0.8248
Epoch 7/10
675/675 [==============================] - 30s 44ms/step - loss: 0.5536 - accuracy: 0.8064 - val_loss: 0.4653 - val_accuracy: 0.8346
Epoch 8/10
675/675 [==============================] - 30s 44ms/step - loss: 0.4955 - accuracy: 0.8278 - val_loss: 0.4189 - val_accuracy: 0.8559
Epoch 9/10
675/675 [==============================] - 31s 45ms/step - loss: 0.4589 - accuracy: 0.8367 - val_loss: 0.3877 - val_accuracy: 0.8650
Epoch 10/10
675/675 [==============================] - 32s 47ms/step - loss: 0.4149 - accuracy: 0.8582 - val_loss: 0.3598 - val_accuracy: 0.8789
169/169 [==============================] - 2s 15ms/step - loss: 0.3598 - accuracy: 0.8789
Validation Accuracy: 87.89%
```

