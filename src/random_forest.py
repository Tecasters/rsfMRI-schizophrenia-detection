import numpy as np
import csv
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from scipy.stats import randint
from nilearn import plotting
from data_prep import get_paths_and_labels
from group_ica import group_ica, write_to_csv

if __name__ == '__main__':

    source_path = '../COBRE_fmri/'
    csv_path = pd.read_csv(source_path + 'cobre_model_group.csv')
    filepaths, binary_class = get_paths_and_labels(csv_path, source_path)
    csvFC_path = 'nn_FCs.csv'
    # uncomment next two lines if you want to do group ICA and overwrite the saved correlation matrices from group ICA
    # corr_mats = group_ica(filepaths)
    # write_to_csv(corr_mats, csvFC_path)
    csv_corr_mats = []
    corr_mats_in_line = []
    with open(csvFC_path, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            corr_mat = np.array(row[0].strip('][').split(', ')).reshape(20, -1)
            csv_corr_mats.append(corr_mat.astype(np.float32))
            corr_mats_in_line.append(np.array(row[0].strip('][').split(', ')).astype(np.float32))

    X_train, X_test, y_train, y_test = train_test_split(corr_mats_in_line, binary_class, test_size=0.3, random_state=1)
    param_dist = {'n_estimators': randint(50, 500), 'max_depth': randint(5,20)}
    rf = RandomForestClassifier()
    rand_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=10, cv=5)
    rand_search.fit(X_train, y_train)
    best_estimator = rand_search.best_estimator_

    print("Best score:", rand_search.best_score_)
    print("Best hyperparams:", rand_search.best_params_)
    y_predict = best_estimator.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_predict))
    print("Precision:", precision_score(y_test, y_predict))
    print("Recall:", recall_score(y_test, y_predict))
    print("F1-score:", f1_score(y_test, y_predict))
    print("Note: As randomised search CV is used, it is likely the best hyperparams will change each iteration, "
          "unless Numpy random state is set to a fixed value beforehand")
    cf = confusion_matrix(y_test, y_predict)
    ConfusionMatrixDisplay(cf).plot()
    plt.show()
