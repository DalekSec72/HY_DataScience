# -*- coding: utf-8 -*-

# 2021 HYU. CSE
# Taehun Kim <th6424@gmail.com>

import os
import sys
from itertools import combinations


def read_file(path):
    transactions = []
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
            lines = list(map(lambda s: s.rstrip('\n'), lines))
            for line in lines:
                transactions.append(line.split('\t'))

    except FileNotFoundError as e:
        print(e)
        sys.exit()
    except FileExistsError as e:
        print(e)
        sys.exit()

    return transactions


def write_file(data, path=sys.argv[3]):
    with open(path, 'a') as f:
        f.writelines(data)


def filter_by_min_sup(candidate: dict, min_sup_count: float) -> dict:
    result = {}
    for item, count in candidate.items():
        if count >= min_sup_count:
            result[item] = count

    return result


def generate_C1(transactions: list) -> dict:
    C1 = {}
    for trx in transactions:
        for item in trx:
            if item in C1.keys():
                C1[item] += 1
            else:
                C1[item] = 1

    return C1


def self_join(itemset, k):
    # self-join
    if k == 2:
        superset = list(combinations(itemset, k))

    else:
        elements = []
        for item in itemset:
            for element in item:
                if element not in elements:
                    elements.append(element)

        superset = list(combinations(elements, k))

    return superset


def prune(k, itemset, superset, min_sup_count):
    # prune
    subsets, temp = {}, []
    if k == 2:
        for item in itemset:
            temp.append(list([item, ]))
        itemset = temp

    else:
        for item in itemset:
            temp.append(set(item))
        itemset = temp

    for subset in superset:
        count = 0
        for item in list(combinations(subset, k - 1)):
            if k == 2:
                item = list(item)
            else:
                item = set(item)

            if item not in itemset:
                break

            count += 1

        if count == k:
            subsets[(subset)] = 0

    # db와 비교하며 카운트
    for item in subsets:
        for trx in transactions_data:
            if set(item).issubset(set(trx)):
                subsets[item] += 1

    return filter_by_min_sup(subsets, min_sup_count)


def get_support_and_confidence_with_associative_item(itemset: dict, k, db):
    # k는 2부터 들어옴.
    for item_set, frequency in itemset.items():
        itemset_len = k

        # subset 개수 따라 association 구성.
        while itemset_len > 1:
            subset = list(combinations(item_set, itemset_len - 1))
            for item in subset:
                complement = set(item_set) - set(item)

                count = 0
                for trx in db:
                    if set(item).issubset(set(trx)):
                        count += 1

                # A와 B가 모두 있을 확률.
                support = frequency / len(db) * 100

                # A가 있을 때 B가 있을 확률 = A가 있을 때 A, B가 같이 있을 확률.
                confidence = frequency / count * 100

                item = set(map(int, set(item)))
                complement = set(map(int, complement))

                result = '{0}\t{1}\t{2}\t{3}\n'.format(str(item), str(complement), str('%.2f' % round(support, 2)),
                                                       str('%.2f' % round(confidence, 2)))

                write_file(result)

            itemset_len -= 1


if __name__ == '__main__':
    minimum_support = float(sys.argv[1]) / 100
    input_file = sys.argv[2]
    output_file = sys.argv[3]

    try:
        os.remove(output_file)
    except FileNotFoundError:
        pass

    transactions_data = read_file(input_file)

    minimum_support_count = minimum_support * len(transactions_data)

    # L[0] = L1, L[1] = L2 ...
    L = [filter_by_min_sup(generate_C1(transactions_data), minimum_support)]
    k = 2
    while True:
        candidate = self_join(L[k - 2], k)
        candidate = prune(k, L[k - 2], candidate, minimum_support_count)

        get_support_and_confidence_with_associative_item(candidate, k, transactions_data)
        if not candidate:
            exit()

        L.append(candidate)
        k += 1
