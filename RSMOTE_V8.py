# _*_coding:utf-8 _*_
# @Time    :2020/8/17
# @Author  :longhai
# @FileName: RSMOTE.py
# @Software: PyCharm


import random
from sklearn.neighbors import NearestNeighbors
import numpy as np
from collections import Counter
from sklearn.cluster import k_means
from sklearn.linear_model import LogisticRegression
import sklearn.metrics
from sklearn.utils import safe_indexing
from 多分类相对密度调用版 import *
from base_sampler import *
import copy


# Calculate how many majority samples are in the k nearest neighbors of the minority samples
def number_maj(imbalanced_featured_data, minor_feature_data, minor_label, imbalanced_label_data):
    nnm_x = NearestNeighbors(n_neighbors=6).fit(imbalanced_featured_data).kneighbors(minor_feature_data,
                                                                                     return_distance=False)[:, 1:]
    nn_label = (imbalanced_label_data[nnm_x] != minor_label).astype(int)
    n_maj = np.sum(nn_label, axis=1)
    return n_maj


class RSmote:
    """
    class RSMOTE usage is as follows:
    "
    clf = LogisticRegression()
    data_Rsmote = RSMOTE.RSmote(data, ir=1, k=5).over_sampling()
    "
    """

    def __init__(self, data, ir=1, k=5, random_state=None):
        """
        :param data: array for all data with label in 0th col.
        :param ir: imbalanced ratio of synthetic data.
        :param k: Number of nearest neighbors.
        """
        self.data = data
        self._div_data()
        self.n_train_less, self.n_attrs = self.train_less.shape
        self.IR = ir
        self.k = k
        self.new_index = 0
        self.random_state = random_state
        self.N = 0
        self.synthetic = None

    def _div_data(self):
        """
        divide the dataset.
        :return: None
        """
        count = Counter(self.data[:, 0])
        a, b = set(count.keys())
        self.tp_less, self.tp_more = (a, b) if count[a] < count[b] else (b, a)

        data_less = self.data[self.data[:, 0] == self.tp_less]
        data_more = self.data[self.data[:, 0] == self.tp_more]

        self.train_less = data_less
        self.train_more = data_more

        self.train = np.vstack((self.train_more, self.train_less))

    def over_sampling(self):
        if self.k + 1 > self.n_train_less:
            print('Expected n_neighbors <= n_samples,  but n_samples = {}, n_neighbors = {}, '
                  'has changed the n_neighbors to {}'.format(self.n_train_less, self.k + 1, self.n_train_less))
            self.k = self.n_train_less - 1
        data_less_filter = []
        num_maj_filter = []
        length_less = len(self.train_less)
        num_maj = number_maj(self.train[:, 1:], self.train_less[:, 1:], self.tp_less, self.train[:, 0])
        for m in range(len(num_maj)):
            if num_maj[m] < self.k:
                data_less_filter.append(self.train_less[m])
                num_maj_filter.append(num_maj[m])
        self.train_less = np.array(data_less_filter)
        distance_more, nn_array_more = NearestNeighbors(n_neighbors=self.k + 1).fit(self.train_more[:, 1:]).kneighbors(
            self.train_less[:, 1:], return_distance=True)
        distance_less, nn_array = NearestNeighbors(n_neighbors=self.k + 1).fit(self.train_less[:, 1:]).kneighbors(
            self.train_less[:, 1:], return_distance=True)

        distance_less = distance_less.sum(axis=1)
        distance_more = distance_more.sum(axis=1)
        distance = distance_less / distance_more
        # print(distance)
        density = 1 / distance  # calculate density

        density = list(map(lambda x: min(100, x), density))  # Control the maximum density range at 100

        # The density is sorted below, and the minority samples are also sorted in order of density.
        density_sorted = sorted(range(len(density)), key=lambda a: density[a], reverse=True)  # sorted
        data_resorted = []
        density_sorted_data = []
        num_sorted = []
        for i in range(len(self.train_less)):
            data_resorted.append(self.train_less[density_sorted[i]])
            density_sorted_data.append(density[density_sorted[i]])
            num_sorted.append(num_maj_filter[density_sorted[i]])

        density = np.array(density_sorted_data)
        cluster_big_density = []
        cluster_small_density = []
        cluster_big_data = []
        cluster_small_data = []
        cluster_big_num = []
        cluster_small_num = []
        cluster = k_means(X=density.reshape((len(density), 1)), n_clusters=2)
        for i in range(cluster[1].shape[0]):
            if cluster[1][i] != cluster[1][i + 1]:  # Partition cluster
                cluster_big_density = density[:i + 1]
                cluster_big_data = np.array(data_resorted)[:i + 1, :]
                cluster_big_num = num_sorted[:i + 1]
                cluster_small_density = density[i + 1:]
                cluster_small_data = np.array(data_resorted)[i + 1:, :]
                cluster_small_num = num_sorted[i + 1:]
                break

        # If there is only one point in a cluster, do not divide the cluster
        if len(cluster_big_data) < 2 or len(cluster_small_data) < 2:
            cluster_big_data = np.array(data_resorted)
            cluster_big_density = density
            cluster_big_num = num_sorted
            flag = 1  # if flag==1 only run big cluster once
        else:
            flag = 2
        sum_0 = 0
        sum_1 = 0
        # Calculate weight
        for p in range(len(cluster_big_num)):
            sum_0 += (5 - cluster_big_num[p]) / self.k + 1
        for p in range(len(cluster_small_num)):
            sum_0 += (5 - cluster_small_num[p]) / self.k + 1

        ratio = []  # save the every cluster's totol weight
        ratio.append(sum_0)
        ratio.append(sum_1)
        wight = [5 / 6, 4 / 6, 3 / 6, 2 / 6, 1 / 6]
        kk = self.k
        diff = len(self.train_more) - length_less  # the number of samples need to synthesize
        totol_less = len(self.train_less)

        for i in range(flag):
            if i == 0:  # big cluster
                density = cluster_big_density
                self.n_train_less = len(cluster_big_data)
                self.train_less = cluster_big_data
                maj_num_ab = cluster_big_num
            else:  # small cluster
                density = cluster_small_density
                self.n_train_less = len(cluster_small_data)
                self.train_less = cluster_small_data
                maj_num_ab = cluster_small_num

            self.k = min(len(self.train_less) - 1, kk)  # if len(self.train_less)<k,set k =len(self.train_less)

            # The number of sample points that need to be inserted at each point
            if flag == 1:
                number_synthetic = int(len(self.train_more) / self.IR - len(self.train_less))
            else:
                if i == 0:
                    number_synthetic = int((len(self.train_less) / totol_less) * diff)
                    len_big = number_synthetic
                else:
                    number_synthetic = diff - len_big

            # Calculate how many points should be inserted for each sample
            N = list(map(lambda x: int((x / ratio[i]) * number_synthetic), wight))
            self.reminder = number_synthetic - sum(N)
            self.num = 0

            neighbors = NearestNeighbors(n_neighbors=self.k + 1).fit(self.train_less[:, 1:])
            nn_array = neighbors.kneighbors(self.train_less[:, 1:], return_distance=False)

            self.synthetic = np.zeros((number_synthetic, self.n_attrs - 1))
            for p in range(self.train_less.shape[0]):
                self._populate(p, nn_array[p][1:], number_synthetic, N, maj_num_ab)

            label_synthetic = np.array([self.tp_less] * number_synthetic).reshape((number_synthetic, 1))
            np.random.seed(self.random_state)
            synthetic_dl = self.synthetic
            synthetic_dl = np.hstack((label_synthetic, synthetic_dl))  # class column

            data_res = synthetic_dl
            if i == 0:
                return_data = np.vstack((copy.deepcopy(self.train), data_res))
                if flag == 1:
                    return return_data
                self.new_index = 0
            else:
                return_data = np.vstack((copy.deepcopy(return_data), data_res))

                return return_data

    # for each minority class samples, generate N synthetic samples.
    def _populate(self, index, nnarray, number_synthetic, N, maj_num_ab):
        random.seed(self.random_state)
        if self.num < self.reminder:
            turn = N[maj_num_ab[index]] + 1
        else:
            turn = N[maj_num_ab[index]]
        for j in range(turn):
            if self.new_index < number_synthetic:
                if self.k == 1:
                    nn = 0
                else:
                    nn = random.randint(0, self.k - 1)
                dif = self.train_less[nnarray[nn], 1:] - self.train_less[index, 1:]
                gap = random.random()
                self.synthetic[self.new_index] = self.train_less[index, 1:] + gap * dif
                self.new_index += 1
            else:
                break
        self.num += 1


class RSmoteKClasses:
    def __init__(self, ir=1, k=5, random_state=None):
        self.ir = ir
        self.k = k
        self.random_state = random_state

    def fit_resample(self, X, y):
        data = np.hstack((y.reshape((len(y), 1)), X))
        counter = Counter(y)

        max_class_label, max_class_number = 0, 0
        for k, v in counter.items():
            if v > max_class_number:
                max_class_label, max_class_number = k, v

        data_new = np.array([]).reshape((-1, data.shape[1]))

        data_more = data[data[:, 0] == max_class_label, :]
        for k, v in counter.items():
            if v == max_class_number:
                continue
            data_less = data[data[:, 0] == k, :]
            data_train = np.vstack((data_more, data_less))
            r_smote = RSmote(data_train, random_state=self.random_state)
            data_r_smote = r_smote.over_sampling()
            if data_new.shape[0] == 0:
                data_new = np.vstack((data_new, data_r_smote))
            else:
                data_new = np.vstack((data_new, data_r_smote[data_r_smote[:, 0] != max_class_label, :]))

        X_resampled, y_resampled = data_new[:, 1:], data_new[:, 0]

        return X_resampled, y_resampled
