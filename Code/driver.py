import pandas as pd
import os
import numpy as np
from classifier import get_classifier, test_clf
from data_preprocessing import normalize_data
from data_sample import over_sample, under_sample, smote_sampling


if __name__ == '__main__':
    cwd = os.getcwd()
    file_path = os.path.join(os.path.dirname(cwd), 'Dataset', 'customer churn', 'features.csv')
    df = pd.read_csv(file_path)

    print(df.head())
    df.drop('Unnamed: 0', axis=1, inplace=True)
    columns = df.columns.values.tolist()
    print(columns)

    test_data = df.sample(frac=0.20)
    df = df.loc[~df.index.isin(test_data.index)]

    print(test_data.shape)
    print(df.shape)

    print(test_data.head())
    print(df.head())


    Y_columns = 'binary_Churn'
    columns.remove(Y_columns)
    print(columns)
    X_columns = columns

    X = df[X_columns]

    scaler, X_scaled = normalize_data(X.values.tolist(), (0, 1))
    df[X_columns] = X_scaled

    # maj_data = df.loc[df['binary_Churn'] == 0]
    # min_data = df.loc[df['binary_Churn'] == 1]
    #
    # maj_data_list = maj_data.values.tolist()
    # min_data_list = min_data.values.tolist()
    #
    # print(len(maj_data_list))
    # print(len(min_data_list))
    #
    # under_sample_maj = under_sample(maj_data_list, len(min_data_list))
    # over_sample_min = over_sample(min_data_list, len(maj_data_list))
    #
    # new_data = under_sample_maj
    # new_data = over_sample_min
    # new_data.extend(min_data_list)
    # new_data.extend(maj_data_list)
    # new_df = pd.DataFrame(new_data, columns=columns)
    # print(new_df)
    # print(new_df['binary_Churn'].value_counts())

    # maj_data_x = maj_data[X_columns].values.tolist()
    # maj_data_y = maj_data[Y_columns].values.tolist()

    # min_data_x = min_data[X_columns].values.tolist()
    # min_data_y = min_data[Y_columns].values.tolist()

    print(df[Y_columns].value_counts())

    X = df[X_columns].values.tolist()
    Y = df[Y_columns].values.tolist()

    # scaled_x = normalize_data(X, (0, 1))

    print(df.shape)

    print(len(X))
    print(len(Y))

    X_res, Y_res = smote_sampling(X, Y)

    X_test = df[X_columns].values.tolist()
    Y_test = df[Y_columns].values.tolist()

    from collections import Counter
    print(Counter(Y_res))

    # clf = get_classifier('RF', [None, 40])
    # Y_list = list(map(lambda x: [x], Y_res))
    # clf.fit(X_res, Y_list)
    # test_clf(clf, X_test, Y_test)

    clf = get_classifier('CS')
    cost_mat = np.zeros((len(X_res), 4))
    # 0 - tn, 1 - fp, 2 - fn, 3 - tp
    cost_mat[:, 0] = 1.5
    cost_mat[:, 1] = 0.5
    clf.fit(X_res, Y_res, cost_mat)
    test_clf(clf, np.array(X_test), np.array(Y_test))
