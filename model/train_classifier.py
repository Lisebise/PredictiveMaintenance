import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier

def load_data(database_filepath):
    """Load data into the needed arrays for the mdoel, messages and categories
    :param database_filepath: (string) filepath of database
    :return:
        - X - (np.array) type of failure to predict
        - Y - (np.array)
        - categories - (np.array) containing all possible categories
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('PredicitiveMaintentace', engine)

    # Y what we want to predict: failure or not
    Y = df["Failure Type"].values
    # all possible categories in Y
    categories = np.unique(Y)

    # drop the not needed columns for X
    df = df.drop(columns=["Failure Type", "Target"])
    X = df.values

    return X, Y, categories


def build_model(classifier):
    """Build model that can be fitted and later on predict values
    :return cv: (GridSearchCV) grid search model with which the training can be done
    """
    classifier_dict = {"LinearCSV": LinearSVC(), "RandomForest": RandomForestClassifier()}
    pipeline = Pipeline([
        ("enc", OneHotEncoder(handle_unknown = "ignore")),
        ("clf", classifier_dict[classifier])
    ])

    parameters = {
        "LinearCSV":{
            "clf__max_iter": [1000, 1500],
            "clf__loss": ["hinge", "squared_hinge"]},
            "clf__class_weight": [{"No Failure": 1, "Other Failure": 2, "Power Failure": 8}],
        "RandomForest":{
            "clf__n_estimators": [100, 200],
            "clf__class_weight": [{"No Failure": 1, "Other Failure": 2, "Power Failure": 30}],
        }
    }

    cv = GridSearchCV(pipeline, parameters[classifier], cv=2, n_jobs=-1, verbose=3)
    return cv


def evaluate_model(model, X_test, Y_test, clf):
    """Evaluate the model based on the accuracy and the classification report
    :param model: (sklearn model) sklearn model that can predict
    :param X_test: (np.array) messgaes that will be categorized
    :param Y_test: (np.array) categories of the X_test
    :param model: (string) string of the classifier name
    :return accuracy: (float) accuracy of trained model
    """
    # predict on test data
    Y_pred = model.predict(X_test)
    accuracy = (Y_pred == Y_test).mean()
    print("Accuracy:", accuracy)
    print("Categories: ", np.unique(Y_pred[:][:]))
    cm = confusion_matrix(Y_test, Y_pred)
    print("\nConfusion Matrix: \n", confusion_matrix(Y_test, Y_pred))

    best_est = model.best_estimator_
    features_in = best_est["clf"].n_features_in_

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = best_est.classes_)
    disp.plot()
    plt.savefig(clf + "confusion_matrix.jpg", bbox_inches='tight')
    return accuracy

def save_model(model, model_filepath):
    """Save the model as pickle file
    :param model: (sklearn model) Trained model that will be saved
    :param model_filepath: (string) string where the model will be saved
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    database_filepath = "C:/Users/Lisann/Documents/Studium/2022_2023 WS RWTH/Data Science Course/" \
                        "CapstoneProject/data/PredicitiveMaintentace"
    models = ["LinearCSV", "RandomForest"]

    # keep track of accuracies
    accuracies = {}
    for clf in models:
        model_filepath = clf
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model(clf)

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        accuracies[clf] = [evaluate_model(model, X_test, Y_test, clf)]

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    # comparing accuracies in a table
    acc_df = pd.DataFrame.from_dict(accuracies)
    acc_df.to_csv("accuracies.csv")

if __name__ == '__main__':
    main()