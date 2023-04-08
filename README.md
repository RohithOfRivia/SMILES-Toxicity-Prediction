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

1. Train the Random Forest Classifier on the training set using the scikit-learn library's 'RandomForestClassifier()' function.
2. Tune the hyperparameters of the Random Forest Classifier using Grid Search Cross-Validation.
3. Evaluate the performance of the Random Forest Classifier on the testing set using several performance metrics, including accuracy, precision, recall, F1-score, and area under the receiver operating characteristic curve (AUC-ROC).
4. Interpret the Random Forest Classifier model by visualizing the feature importance using the Matplotlib library.

## Requirements

The notebook requires the following libraries to be installed:

- Pandas
- scikit-learn
- RDKit
- Matplotlib
- Jupyter

## Usage

To run the notebook, you can do the following:

1. Install the required libraries using pip:
