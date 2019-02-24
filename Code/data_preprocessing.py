from sklearn.preprocessing import MinMaxScaler


def normalize_data(data_X, feature_range):
    scaler = MinMaxScaler(feature_range=feature_range)
    normalized_X = scaler.fit_transform(data_X)
    return scaler, normalized_X
