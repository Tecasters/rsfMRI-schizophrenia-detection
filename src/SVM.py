from nilearn import datasets
from pathlib import Path
import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.decomposition import PCA
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import pickle
import matplotlib.pyplot as plt
from data_prep import get_paths_and_labels


def get_corr_matrix(fmri_path, atlas, resolution='064'):

    scale = atlas['scale'+resolution]
    masker = NiftiLabelsMasker(labels_img=scale, standardize=False)
    time_series = masker.fit_transform(fmri_path)

    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([time_series])[0]

    np.fill_diagonal(correlation_matrix, 0.9999)
    fisher = np.arctanh(correlation_matrix)

    nan_mean = np.nanmean(fisher)
    fisher = np.where(np.isnan(fisher), nan_mean, fisher)

    return fisher


def extract_features(corr_matrices):
    stacked = np.vstack(corr_matrices)

    n_components = min(stacked.shape) - 1

    pca = PCA(n_components=n_components)
    features = pca.fit_transform(stacked)
    return features, pca.components_


def classifierSVM(features, labels):

    X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                        test_size=0.3,
                                                        random_state=0)
    classifier = svm.SVC()
    space = dict()
    space['C'] = [1, 10, 100, 1000]
    space['kernel'] = ['rbf']
    space['gamma'] = [0.001, 0.0001]
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    search = GridSearchCV(classifier, space, scoring='accuracy', cv=cv)
    result = search.fit(X_train, y_train)

    now = datetime.now()
    time_str = now.strftime("%d_%m_%Y_%H%M%S")
    path = "./models/svm_" + time_str + ".pth"
    pickle.dump(classifier, open(path, "wb"))
    # model = pickle.load(open(path, "rb"))
    print("Best score:", result.best_score_)
    print("Best hyperparams:", result.best_params_)
    y_predict = result.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_predict))
    print("Precision:", precision_score(y_test, y_predict))
    print("Recall:", recall_score(y_test, y_predict))
    print("F1-score:", f1_score(y_test, y_predict))
    print("Confusion matrix:")
    cf = confusion_matrix(y_test, y_predict, labels=[0.0, 1.0])
    disp = ConfusionMatrixDisplay(cf, display_labels=['HC', 'SZ'])
    disp.plot()
    plt.show()


if __name__ == '__main__':

    atlas = datasets.fetch_atlas_basc_multiscale_2015(data_dir=Path('..'))
    source_path = '../COBRE_fmri/'
    csv = pd.read_csv(source_path + 'cobre_model_group.csv')
    filepaths, binary_class = get_paths_and_labels(csv, source_path)
    resolutions = ['007', '036', '064', '122', '444']
    for resolution in resolutions:
        starttime = datetime.now()
        corr_matrices = []
        labels = []
        for i in range(len(binary_class)):
            print("Calculating correlation for ", i)
            corr_matrices.append(get_corr_matrix(filepaths[i], atlas, resolution))

        features, pca_components = extract_features(corr_matrices)

        print(features.shape)
        features = features.reshape((146, -1))
        print(features.shape)

        classifierSVM(features, binary_class)
        print("Resolution:", resolution, "Time:", datetime.now()-starttime)