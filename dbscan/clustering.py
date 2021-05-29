# -*- coding: utf-8 -*-

# 2021 HYU. CSE
# Taehun Kim <th6424@gmail.com>

import os
import sys
import time
import numpy as np

from dbscan import DBSCAN


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
        exit()
    except FileExistsError as e:
        print(e)
        exit()
    finally:
        f.close()

    data = np.array(samples)

    return data


def sort_clusters_by_size(labels):
    cluster, counts = np.unique(labels, return_counts=True)
    cluster_count_dict = dict(zip(cluster, counts))
    try:
        del cluster_count_dict[-1]
    except KeyError:
        pass

    return sorted(cluster_count_dict.items(), key=lambda x: x[1], reverse=True)


def write_file(path, data):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass

    with open(path, 'a') as f:
        f.writelines(data)


def write_result_files(input_file_name, n, result, labels):
    for i in range(0, min(n, len(result))):
        output_file_path = f'{input_file_name[:6]}_cluster_{i}.txt'
        output = []
        for idx, label in enumerate(labels):
            if label == result[i][0]:
                line = f'{idx}\n'
                output.append(line)

        write_file(output_file_path, output)


if __name__ == '__main__':
    start_time = time.time()
    print('프로그램 가동')
    input_file = sys.argv[1]
    n = int(sys.argv[2])
    eps = int(sys.argv[3])
    min_pts = int(sys.argv[4])

    data = read_file(input_file)

    dbscan = DBSCAN(eps, min_pts)
    labels = dbscan.run(data)

    sorted_dict = sort_clusters_by_size(labels)

    write_result_files(input_file, n, sorted_dict, labels)

    print(f'프로그램 종료')
    print(f'프로그램 가동 시간: {time.time() - start_time}')
