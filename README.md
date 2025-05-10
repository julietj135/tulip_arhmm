# Outer directory files

**run.py**

This file contains functions that allow us to train and use the AR-HMM. Functions include editing configuration files, performing pca, getting augmented data, training the model and saving the results, and applying the model on new data.

**test.py**

Can ignore, was just to test code on DCC.

**work.py**

Execution script that loads in data and applies all other functions.

# Subdirectories

## Preprocessing

### Internal directories
**configs**

Contains all configuration files for each behaviorial test.

**dappy and ssumo**

Josh's packages that has inverse and forward kinematic functions.

# Files
**augment.py**

For augmenting fingertapping and gait behavioral data by using a sliding window.

**data.py**

Contains functions for loading in and formatting data. Standardizes data with functions from **kinematic_processing.py**

**kinematic_processing.py**

Contains functions for performing inverse and forward kinematics.

## Keypointmoseq
Keypoint MoSeq with some edits for figure making, 6D coordinate training, and behavioral test specifications.

## Analysis

**classifier.py**

Script with functions for binary and multiclass classification with XGBoost and logistic regression. Performs cross validation and hyperparameter tuning. Uses functions from **features.py** for feature engineering.

**features.py**

Extracts features from syllable data, such as frequency and duration statistics.

**results.py**

Functions for plots and videos of individual models (e.g. syllable gifs and videos, transition matrices, etc.).

**scans.py**

Functions for plots across an ensemble of models (e.g. EML score plots).