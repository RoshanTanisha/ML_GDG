import numpy
from imblearn.over_sampling import SMOTE
import random

random.seed(42)


def under_sample(majority_data: list, minority_data_len: int) -> list:
    under_sampled_majority_data = random.sample(majority_data, minority_data_len)
    return under_sampled_majority_data


def over_sample(minority_data: list, majority_data_len: int) -> list:
    over_sampled_minority_data = minority_data
    len_new_data = len(over_sampled_minority_data)
    list_indices = range(len(minority_data))

    while len_new_data < majority_data_len:
        index = random.choice(list_indices)
        over_sampled_minority_data.append(minority_data[index])
        len_new_data += 1

    return over_sampled_minority_data


def smote_sampling(x, y):
    sampler = SMOTE(sampling_strategy='not majority', random_state=42, k_neighbors=2, n_jobs=-1)
    x_res, y_res = sampler.fit_resample(x, y)
    return x_res, y_res


if __name__ == '__main__':
    a, b = smote_sampling([[1,2,3], [4,5,6], [7,8,9], [0,9,0]], [1,1,0,0])
    c = numpy.append(a.transpose(), [b], axis=0).transpose()
