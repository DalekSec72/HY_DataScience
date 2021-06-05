# -*- coding: utf-8 -*-

# 2021 HYU. CSE
# Taehun Kim <th6424@gmail.com>

import os
import sys

import numpy as np
import pandas as pd


def read_file(path):
    try:
        if not os.path.isfile(path):
            raise FileExistsError
    except FileExistsError as e:
        print(e)
        sys.exit()

    cols = ['user', 'item', 'rating', 'timestamp']
    dataframe = pd.read_csv(path, sep='\t', names=cols)
    # 타임스탬프 필요없으니 드롭
    dataframe.drop(columns='timestamp', inplace=True)
    # 유저를 인덱스, 아이템을 컬럼, 레이팅을 밸류로.
    dataframe = dataframe.pivot(index='user', columns='item', values='rating')

    return dataframe


class MatrixFactorization():
    def __init__(self, R, k, learning_rate, reg_param, epochs, verbose=False):
        """
        :param R: rating matrix
        :param k: latent parameter
        :param learning_rate: alpha on weight update
        :param reg_param: beta on weight update
        :param epochs: training epochs
        """

        self._R = R
        self._num_users, self._num_items = R.shape
        self._k = k
        self._learning_rate = learning_rate
        self._reg_param = reg_param
        self._epochs = epochs

    def fit(self):
        # 쪼개지는 매트릭스 P, Q
        self._P = np.random.normal(size=(self._num_users, self._k))
        self._Q = np.random.normal(size=(self._num_items, self._k))

        # biases
        self._b_P = np.zeros(self._num_users)
        self._b_Q = np.zeros(self._num_items)
        self._b = np.mean(self._R[np.where(self._R != 0)])

        # n epochs 학습
        self._training_process = []
        for epoch in range(self._epochs):
            for i in range(self._num_users):
                for j in range(self._num_items):
                    # 레이팅이 있는 셀 학습
                    if self._R[i, j] > 0:
                        self.gradient_descent(i, j, self._R[i, j])
            cost = self.cost()
            self._training_process.append((epoch, cost))

            if epoch % 10 == 0:
                print(f"epoch: {epoch} ; cost = {cost: .4f}")

        self.print_results()

    def cost(self):
        xi, yi = self._R.nonzero()
        predicted = self.get_complete_matrix()
        cost = 0
        for x, y in zip(xi, yi):
            cost += (self._R[x, y] - predicted[x, y]) ** 2
        return np.sqrt(cost / len(xi))

    def gradient(self, error, i, j):
        dp = (error * self._Q[j, :]) - (self._reg_param * self._P[i, :])
        dq = (error * self._P[i, :]) - (self._reg_param * self._Q[j, :])
        return dp, dq

    def gradient_descent(self, i, j, rating):
        # get error
        prediction = self.get_prediction(i, j)
        error = rating - prediction

        # update biases
        self._b_P[i] += self._learning_rate * (error - self._reg_param * self._b_P[i])
        self._b_Q[j] += self._learning_rate * (error - self._reg_param * self._b_Q[j])

        # update latent feature
        dp, dq = self.gradient(error, i, j)
        self._P[i, :] += self._learning_rate * dp
        self._Q[j, :] += self._learning_rate * dq

    def get_prediction(self, i, j):
        return self._b + self._b_P[i] + self._b_Q[j] + self._P[i, :].dot(self._Q[j, :].T)

    def get_complete_matrix(self):
        return self._b + self._b_P[:, np.newaxis] + self._b_Q[np.newaxis:, ] + self._P.dot(self._Q.T)

    def print_results(self):
        print("Final R matrix:")
        print(self.get_complete_matrix())
        print("Final RMSE:")
        print(self._training_process[self._epochs - 1][1])


if __name__ == "__main__":
    R = np.array(read_file('data-2/u1.base').fillna(0))

    factorizer = MatrixFactorization(R, k=1000, learning_rate=0.001, reg_param=0.01, epochs=50)
    factorizer.fit()
