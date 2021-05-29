# -*- coding: utf-8 -*-

# 2021 HYU. CSE
# Taehun Kim <th6424@gmail.com>

import os
import sys
import pandas as pd
import node

sys.setrecursionlimit(10**6)

def read_file(path):
    samples = []
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
            lines = list(map(lambda s: s.rstrip('\n'), lines))
            for line in lines:
                samples.append(line.split('\t'))

    except FileNotFoundError as e:
        print(e)
        sys.exit()
    except FileExistsError as e:
        print(e)
        sys.exit()

    dataframe = pd.DataFrame(samples)

    dataframe.columns = dataframe.iloc[0]
    dataframe.drop(index=0, inplace=True)

    return dataframe


def write_file(path, data):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass

    with open(path, 'a') as f:
        f.writelines(data)


if __name__ == '__main__':
    train_file_path = sys.argv[1]
    test_file_path = sys.argv[2]
    result_file_path = sys.argv[3]

    train_data = read_file(train_file_path)
    test_data = read_file(test_file_path)

    decision_tree = node.Node(train_data)

    output = []

    temp = ''
    for item in train_data.columns.to_list():
        temp += item + '\t'
    output.append(temp + '\n')
    # iterrows 의 퍼포먼스는 끔찍하다.
    count = 0
    for index, row in test_data.iterrows():
        label = decision_tree.test(row)

        temp = ''
        for item in row.to_list():
            temp += item + '\t'
        temp += label + '\n'
        output.append(temp)
        print(count, temp)
        count += 1

    write_file(result_file_path, output)
