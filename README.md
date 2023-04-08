# SMILES Toxicity Prediction

This notebook contains code for predicting the toxicity of chemical compounds using their Simplified Molecular Input Line Entry System (SMILES) representation.

## Dataset

The dataset used for this project is the The Toxicity Prediction Challenge II dataset, which contains chemical compounds represented as SMILES strings, along with their respective assay ID and toxicity labels. The dataset was obtained from [Kaggle](https://www.kaggle.com/competitions/the-toxicity-prediction-challenge-ii/overview).

## Data Preparation

The notebook performs the following steps to prepare the data for modeling:

1. Load the dataset and pass through the pipeline to get all descriptors and morgan fingerprints for each compound
2. Performs data cleaning
3. 

## Modeling

The notebook uses a Random Forest Classifier as the machine learning model to predict the toxicity of chemical compounds. The following steps are performed for modeling:

1. Train the XGBoost Classifier Classifier on the training set using the xgboost library's 'XGBClassifier()' function. Hyperparameters: n_estimators=700, max_depth=10
2. Evaluate the performance of the model on the testing set using F1-score performance metric.

## Requirements

The notebook requires the following libraries to be installed:

- Pandas
- scikit-learn
- RDKit
- Matplotlib
- Jupyter
- xgboost

## Usage

## Google Collab

To run the notebook, you can do the following:

1. Upload the notebook to Gogle [Collab](https://www.kaggle.com/competitions/the-toxicity-prediction-challenge-ii/overview](https://colab.research.google.com/).
2. Select Runtime, and then select Run all in the dropdown menu. A csv file containing all the predicitons made by the model on the test data will be downloaded directly on the browser after all cells finish executing. 
