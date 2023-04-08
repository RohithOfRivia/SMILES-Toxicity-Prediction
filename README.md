# SMILES Toxicity Prediction

This notebook contains code for predicting the toxicity of chemical compounds using their Simplified Molecular Input Line Entry System (SMILES) representation.

## Dataset

The dataset used for this project is the The Toxicity Prediction Challenge II dataset, which contains chemical compounds represented as SMILES strings, along with their respective assay ID and toxicity labels. The dataset was obtained from [Kaggle](https://www.kaggle.com/competitions/the-toxicity-prediction-challenge-ii/overview).

## Data Preparation

The notebook performs the following steps to prepare the data for modeling:

1. Load the dataset and pass through the pipeline to get all descriptors and morgan fingerprints for each compound
2. Performs data cleaning
3. Model training
4. Writes all predictions made on test data to a csv file 

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

### Kaggle Container

[This](https://www.kaggle.com/x2022gvu/x2022-gvu-best-score) container for the notebook hosted in Kaggle can be run using Docker.

Option 1: Go to Logs section to find the latest container image, and then run image using docker.

Option 2: Download Notebook from the link and run all cells to get the cv file for predictions as output.

Note: The notebook outputs the cv file with an index column, Id, Predicted. If the identical submission cv file is required, the index column has to be dropped. This can be easily done by passing "index-False* argument in the 2nd cell of code in the Saving predictions section.

### Google Collab

To run the notebook, you can do the following:

1. Download a copy of the [notebook](https://www.kaggle.com/x2022gvu/x2022-gvu-best-score) from the options.
2. Upload the notebook to Google [Collab](https://colab.research.google.com/).
3. Scroll down to the last section called "Saviong predictions for submission", delete or comment the content in the second cell in that section, which has one line of code 0 "res.to _csv".
4. Uncomment every line starting at the line "from google.colab...
5. Select Runtime from the options menu at the top, and then select Run all in the dropdown menu. A csv file containing all the predicitons made by the model on the test data will be downloaded directly through the browser after all cells finish executing.
