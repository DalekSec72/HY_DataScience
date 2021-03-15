# -*- coding: utf-8 -*-

# 2021 HYU. CSE
# Taehun Kim <th6424@gmail.com>

import sys
from itertools import combinations


def read_data(path):
    transactions = []
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
            lines = list(map(lambda s: s.rstrip('\n'), lines))
            for line in lines:
                transactions.append(line.split('\t'))

    except FileNotFoundError as e:
        print(e)
        exit()
    except FileExistsError as e:
        print(e)
        exit()
    finally:
        f.close()

    return transactions


def filter_by_min_sup(candidate: dict, min_sup_count: float) -> list:
    result = []
    for item in candidate:
        if candidate[item] >= min_sup_count:
            result.append(item)

    return result if result else exit()


def generate_C1(transactions: list) -> dict:
    C1 = {}
    for trx in transactions:
        for item in trx:
            if item in C1.keys():
                C1[item] += 1
            else:
                C1[item] = 1

    return C1


def generate_candidate(itemset: list, k) -> dict:
    # self-join
    if k == 2:
        for item in itemset:
            superset = list(combinations(itemset, k))

    else:
        elements = []
        for item in itemset:
            for element in item:
                if element not in elements:
                    elements.append(element)
        superset = list(combinations(elements, k))

    # prune
    subsets, temp = [], []
    if k == 2:
        for item in superset:
            subsets.append(list(combinations(item, k - 1)))

        for subset in subsets:
            if subset[0][0] in itemset and subset[1][0] in itemset:
                temp.append((subset[0][0], subset[1][0]))

    else:
        for item in superset:
            subsets.append(list(combinations(item, k - 1)))

        index = -1
        for subset in subsets:
            index += 1
            count = 0

            for element in subset:
                if set((element,)).issubset(set(itemset)):
                    count += 1

            if count == len(subset):
                temp.append(superset[index])

    candidate = {}
    # db와 비교하며 카운트
    for item in temp:
        for trx in transactions_data:
            if set(item).issubset(set(trx)):
                if item in candidate.keys():
                    candidate[item] += 1
                else:
                    candidate[item] = 1

    return candidate if candidate else exit()


if __name__ == '__main__':
    minimum_support = float(sys.argv[1]) / 100
    input_file = sys.argv[2]
    output_file = sys.argv[3]

    global transactions_data
    transactions_data = read_data(input_file)

    minimun_support_count = minimum_support * len(transactions_data)

    L = [filter_by_min_sup(generate_C1(transactions_data), minimum_support)]
    # L[0] = L1, L[1] = L2 ...

    k = 0
    while True:
        print('L_', f'{k + 1}', L[k])
        L.append(filter_by_min_sup(generate_candidate(L[k], k+2), minimun_support_count))
        k += 1
