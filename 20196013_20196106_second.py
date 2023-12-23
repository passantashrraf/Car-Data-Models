import pandas as pd

import matplotlib.pyplot as plt

import numpy as np
from sklearn import linear_model


def path_to_df(path):
    """takes path to csv file and returns a pandas dataframe"""
    df = pd.read_csv(path)
    return df


def normalize(df):
    "normalizes the df based on the min-max technique"
    df.iloc[:, 0:2] = (df.iloc[:, 0:2] - df.iloc[:, 0:2].min()) / \
        (df.iloc[:, 0:2].max() - df.iloc[:, 0:2].min())
    return df


def make_scatter_plot(df):

    # create scatter plot
    colors = ['red' if x == 0 else 'blue' for x in df['purchased']]
    plt.scatter(df['age'], df['salary'], c=colors, cmap='cool', alpha=0.8)

    plt.xlabel('age')
    plt.ylabel('salary')
    plt.title('Scatter Plot of age and salary')

    plt.show()


def split_data(df, test_fraction):
    "splits data using indexing"
    # Calculate the number of samples for testing
    num_test = int(np.ceil(test_fraction * len(df)))
    # Split the dataframe into training and testing sets
    train_df = df[num_test:]
    test_df = df[:num_test]
    return train_df, test_df


def seperate_data(df):
    "returns X (age & salary) and Y (purchased)"
    y = df['purchased'].to_numpy()
    x = df.drop(columns=['purchased']).to_numpy().reshape((-1, 2))
    return x, y


def build_model():
    model = linear_model.LogisticRegression()
    return model


def train_model(model, x_train, y_train):
    model.fit(x_train, y_train)
    print(model.coef_)

    return model


def eval_model(model, x_test, y_test):
    accuracy = model.score(x_test, y_test)
    print(accuracy)


def main():
    path = "customer_data.csv"
    df = path_to_df(path)
    normalized_df = normalize(df)
    make_scatter_plot(normalized_df)
    train, test = split_data(normalized_df, 0.2)
    X_train, y_train = seperate_data(train)
    X_test, y_test = seperate_data(test)
    model = build_model()
    model = train_model(model, X_train, y_train)
    eval_model(model, X_test, y_test)


main()
