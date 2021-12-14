from argparse import ArgumentParser

import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    KBinsDiscretizer
)

parser = ArgumentParser()
parser.add_argument("-i", "--input_location", dest="input",
                    default="s3://datalake-us-east-1/input/census-income.data.csv",
                    help="S3 location for input file")
parser.add_argument("-o", "--output_location", dest="output",
                    default='s3://datalake-us-east-1/output/',
                    help="S3 location for output file")

args = parser.parse_args()


input_data = args.input
output_path = args.output

columns = [
    "age",
    "education",
    "major industry code",
    "class of worker",
    "capital gains",
    "capital losses",
    "dividends from stocks",
    "income",
]
class_labels = [" - 50000.", " 50000+."]

# --------------------
df = pd.read_csv(input_data)
df = df.iloc[:, [0, 4, 8, 1, 16, 17, 18, 41]]
df.columns = columns

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df.replace(class_labels, [0, 1], inplace=True)

negative_examples, positive_examples = np.bincount(df["income"])
print(f"Data after cleaning: {df.shape}, {positive_examples} positive examples, {negative_examples} negative examples")

# --------------------
split_ratio = 0.3
print("Splitting data into train and test sets with ratio {}".format(split_ratio))
X_train, X_test, y_train, y_test = train_test_split(
    df.drop("income", axis=1), df["income"], test_size=split_ratio, random_state=0
)

# --------------------
preprocess = make_column_transformer(
        (KBinsDiscretizer(encode="onehot-dense", n_bins=10), ["age"]),
        (StandardScaler(), ["capital gains", "capital losses", "dividends from stocks"]),
        (OneHotEncoder(sparse=False), ["education", "major industry code", "class of worker"]),
    )
print("Running preprocessing and feature engineering transformations")
train_features = preprocess.fit_transform(X_train)
test_features = preprocess.transform(X_test)

# --------------------
print("Train data shape after preprocessing: {}".format(train_features.shape))
print("Test data shape after preprocessing: {}".format(test_features.shape))
train_features_output_path = output_path + "train/train_features.csv"
train_labels_output_path = output_path + "train/train_labels.csv"

test_features_output_path = output_path + "test/test_features.csv"
test_labels_output_path = output_path + "test/test_labels.csv"

print("Saving training features to {}".format(train_features_output_path))
pd.DataFrame(train_features).to_csv(train_features_output_path, header=False, index=False)

print("Saving test features to {}".format(test_features_output_path))
pd.DataFrame(test_features).to_csv(test_features_output_path, header=False, index=False)

print("Saving training labels to {}".format(train_labels_output_path))
y_train.to_csv(train_labels_output_path, header=False, index=False)

print("Saving test labels to {}".format(test_labels_output_path))
y_test.to_csv(test_labels_output_path, header=False, index=False)
