# -*- coding: utf-8 -*-

# 2021 HYU. CSE
# Taehun Kim <th6424@gmail.com>

import os
import sys
import time
import numpy as np
import pandas as pd
import tensorflow as tf


# Hyper params
learning_rate = 0.001
training_epochs = 3000
feature_len = 100


def read_file(train, test):
    df_train = pd.read_csv(train, sep='\t', names=['user', 'item', 'rate', 'time'])
    df_test = pd.read_csv(test, sep='\t', names=['user', 'item', 'rate', 'time'])

    return df_train, df_test


def write_file(path, data):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass

    with open(path, 'a') as f:
        f.writelines(data)


def adam_optimizer(cost):
    train_step = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-08,
        use_locking=False,
        name='Adam'
    ).minimize(cost)

    return train_step


def build_R_matrix(num_user, num_item):
    # Matrix factorization, R(u, i) = U(u, f) * V(f, i)
    U = tf.Variable(tf.random_uniform([num_user, feature_len]))
    V = tf.Variable(tf.random_uniform([feature_len, num_item]))
    result = tf.matmul(U, V)
    result_flatten = tf.reshape(result, [-1])

    return result, result_flatten


if __name__ == '__main__':
    start_time = time.time()

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    result_file_path = train_file.split('/')[-1] + '_prediction.txt'

    df_train, df_test = read_file(train_file, test_file)
    # df_train = df_train.pivot(index='user', columns='item', values='rate').fillna(2).stack().reset_index(name='rate')

    """ Zero injection
    # rate가 있으면 1 없으면 nan.
    R_pre = df_train.pivot(index='user', columns='item', values='rate')
    R_pre[R_pre.notnull()] = 1
    """

    # 인덱스가 1부터 시작하니 한 칸씩 당겨줌.
    user_indices = [x - 1 for x in df_train.user.values]
    item_indices = [x - 1 for x in df_train.item.values]
    # pre_preference = [1] * df_train.rate.values.size
    rates = df_train.rate.values

    num_user = max(df_train.user.max(), df_test.user.max())
    num_item = max(df_train.item.max(), df_test.item.max())

    result, result_flatten = build_R_matrix(num_user, num_item)

    # rating
    # result_flatten에서 rate를 가져옴. 2차원 행렬에서 (user, item)의 값이 1차원 벡터의 아래 indices 파라미터 값에 해당함.
    R = tf.gather(result_flatten, user_indices * tf.shape(result)[1] + item_indices)

    # SAE
    # cost_pre = tf.reduce_sum(tf.abs(R - pre_preference))
    cost = tf.reduce_sum(tf.abs(R - rates))

    # train_step_pre = adam_optimizer(cost_pre)
    train_step = adam_optimizer(cost)

    # pre_ckpt_path = "output/pre_preference/"
    ckpt_path = f"output/{train_file.split('/')[-1]}/"
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("학습시작")

        for epoch in range(training_epochs):
            c, _ = sess.run([cost, train_step])
            if epoch % 100 == 0:
                print(f"Epoch: {epoch}, cost: {c}")

        saver = tf.train.Saver()
        saver.save(sess, ckpt_path)
        saver.restore(sess, ckpt_path)

        print('Training data restoration...')
        r_hat = np.clip(sess.run(result), 1, 5)

        for u, v, r in df_train[['user', 'item', 'rate']].values[:10]:
            print(f'Rating for user: {str(u)} for item {str(v)}: {str(r)}, prediction: {str(r_hat[u-1][v-1])}')

        print('Test data prediction...')
        output = []
        for u, v, r in df_test[['user', 'item', 'rate']].values:
            line = (str(u) + '\t' + str(v) + '\t' + str(r_hat[u-1][v-1]) + '\n')
            output.append(line)

        write_file(result_file_path, output)
        print(f'Time: {time.time() - start_time}')
