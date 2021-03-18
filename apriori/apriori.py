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
        exit()
    except FileExistsError as e:
        print(e)
        exit()
    finally:
        f.close()

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


def generate_candidate(itemset: dict, k) -> dict:
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
                if {element}.issubset(set(itemset)):
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

    return candidate


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

    i = 0
    while True:
        c = generate_candidate(L[i], i + 2)
        if not c:
            break

        l = filter_by_min_sup(c, minimum_support_count)
        if not l:
            break

        L.append(l)

        i += 1

    for j in range(0, len(L)):
        get_support_and_confidence_with_associative_item(L[j], j + 1, transactions_data)
