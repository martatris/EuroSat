# EuroSAT Land Use and Land Cover Classification
=============================================

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

Main Script
------------
The main Python file is `EuroSat.py`.

Key Steps in the Code:
1. Loads the dataset using `image_dataset_from_directory`.
2. Preprocesses and batches the data.
3. Builds a CNN model.
4. Trains the model on the EuroSAT dataset.
5. Evaluates accuracy and plots results.

Dependencies
-------------
Install these packages before running the script:

    pip install tensorflow matplotlib

Optional for better performance:
    pip install tensorflow-datasets scikit-learn protobuf==3.20.*

Running the Project
-------------------
1. Make sure your data directory path is correctly set in the script:
       DATA_DIR = "/Users/<yourname>/Desktop/Data/EuroSAT"
2. Run the script:
       python EuroSat.py
3. The model will train for several epochs and display accuracy results and training graphs.

Output
------
- Training and validation accuracy per epoch.
- Final validation accuracy printed in the console.
- A plot showing accuracy trends over epochs.

Author
------
Created for educational purposes — Land Use and Land Cover classification using Python and TensorFlow.
