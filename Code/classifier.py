import tflearn
import costcla
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report, accuracy_score
from imblearn.metrics import geometric_mean_score
import numpy as np
import matplotlib.pyplot as plt


def get_classifier(type=None, input_data_shape=None):
    if type == 'RF':
        clf = RandomForestClassifier(n_jobs=-1, n_estimators=10, random_state=0, verbose=1)
    elif type == 'SVM':
        clf = SVC(random_state=0, kernel='poly', verbose=1)
    elif type == 'CRF':
        class_weights = {0: 1.5, 1: 1}
        clf = RandomForestClassifier(n_jobs=-1, n_estimators=10, random_state=0, class_weight=class_weights, verbose=1)
    elif type == 'CSVM':
        class_weights = {0: 1.5, 1: 1}
        clf = SVC(random_state=0, kernel='linear', class_weight=class_weights, verbose=1)
    elif type == 'NN':
        if input_data_shape is None:
            raise Exception('Please provide input data shape')
        net = tflearn.input_data(shape=input_data_shape)
        net = tflearn.fully_connected(net, 100)
        net = tflearn.fully_connected(net, 1, activation='sigmoid')
        net = tflearn.regression(net, optimizer='adam', loss='mean_square')

        model = tflearn.DNN(net)
        clf = model
    elif type == 'CS':
        clf = costcla.CostSensitiveLogisticRegression(verbose=1, max_iter=10)
    else:
        raise Exception('Provide a proper classifier')
    return clf

def test_clf(clf, X_test, Y_test):
    print(clf)
    try:
        y_prob = clf.predict_proba(X_test)[:,1]
        print('ACCURACY = ', accuracy_score(Y_test, np.round(y_prob)))
        print('GEOMETRIC MEAN SCORE = ', geometric_mean_score(Y_test, np.round(y_prob)))
        print(classification_report(Y_test, np.round(y_prob)))
        roc_score = roc_auc_score(Y_test, y_prob)
        fpr, tpr, threshold = roc_curve(Y_test, y_prob)
        plt.plot(fpr, tpr, label='ROAUC')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROAUC SCORE = %.2f' % roc_score)
        plt.legend(loc='best')
        plt.show()
    except Exception as e:
        y_pred = clf.predict(X_test)
        print(y_pred)
        print('ACCURACY = ', accuracy_score(Y_test, np.round(y_pred)))
        print('GEOMETRIC MEAN SCORE = ', geometric_mean_score(Y_test, np.round(y_pred)))
        print(classification_report(Y_test, np.round(y_pred)))

if __name__ == '__main__':
    # classifiers = ['RF', 'SVM', 'CRF', 'CSVM', 'CS']
    # for classes in classifiers:
    #     print(get_classifier(classes))

    print(get_classifier('NN', [None, 20]))