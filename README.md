# preprocessing-data-in-docker
Preprocessing data with arguments of input/output path inside docker container

- remove duplicates and rows with conflicting data
- transform the target income column into a column containing two labels.
- transform the age and num persons worked for employer numerical columns into categorical features by binning them
- scale the continuous capital gains, capital losses, and dividends from stocks so they're suitable for training
- encode the education, major industry code, class of worker so they're suitable for training
- split the data into training and test datasets, and saves the training features and labels
and test features and labels.

The dataset used here is the [Census-Income KDD Dataset](https://archive.ics.uci.edu/ml/datasets/Census-Income+%28KDD%29). You select features from this dataset, 
clean the data, and turn the data into features that the training algorithm can use to train 
a binary classification model, and split the data into train and test sets. 
The task is to predict whether rows representing census responders have an income greater than $50,000,
or less than $50,000. The dataset is heavily class imbalanced, with most records being labeled 
as earning less than $50,000. After training a logistic regression model, you evaluate the model 
against a hold-out test dataset, and save the classification evaluation metrics, including precision, 
recall, and F1 score for each label, and accuracy and ROC AUC for the model.