# -*- coding: utf-8 -*-

# 2021 HYU. CSE
# Taehun Kim <th6424@gmail.com>

import pandas as pd
import math
# from itertools import combinations


class Node:
    is_leaf = False
    children = {}
    data = None
    attribute = ''
    # split_point = []
    class_label = ''

    def __init__(self, data: pd.DataFrame, is_leaf=False):
        self.is_leaf = is_leaf
        self.data = data
        self.children = {}

        # leaf 가 아니면 노드 확장.
        if not self.is_leaf:
            self.expand()

        # leaf 가 아니어도 자식노드 없는 경우 때문에 라벨 갖고 있어야함.
        self.class_label = self.data.iloc[:, -1].value_counts().idxmax()

    # Info(D) = - sigma(i=1 to m) p_i * log(p_i, 2), p_i는 어떤 튜플의 클래스가 i일 확률 = 클래스 C인 개수 / 전체 튜플 수.
    def calculate_entropy(self, row):
        total = row.sum()
        sum = 0
        for val in row:
            # 로그에 0이 들어가면 안됨, 0인 경우는 homogeneous 한 경우.
            if val:
                sum += val / total * math.log(val / total, 2)
            else:
                return 0

        return -sum

    """
    attribute 를 하나 골라 gain ratio 리턴.
    gain 은 전체 데이터셋에 대한 info 에서 attribute A로 나눴을 떄 info 를 뺀 값이지만
    info(D)는 어떤 A를 고르더라도 동일하므로 info_A(D)만 구해서 나눈 info 가 작으면 gain 이 큼.
    따라서 calculate_gain_ratio 에서는
    Gain = (info(D) - info_A(D)) / splitinfo 를 전개하여 (info / split) - (info_A(D) / split) 으로 만든 후
    앞쪽 항을 무시하고 (info_A(D) / split) 에 집중하여 해당 값이 작음 = gain ratio 가 큼 을 이용함.
    """
    def calculate_gain_ratio(self, attribute):
        class_col = self.data.columns[-1]

        # attribute 의 내용이 인덱스, 컬럼에는 라벨 종류별로 해당 라벨에 해당하는 것 개수.
        # shape 는 attribute 의 라벨 수 n, 클래스 종류 m, (n, m)
        df = self.data.groupby([attribute, class_col]).size().unstack().fillna(0)

        # 전체 튜플 수.
        total = df.values.sum()

        # Weighted average Info_A(D) = sigma(j=1 to v) * (D_j / D) * info(D_j)
        sum_info = 0
        for row in df.values:
            sum_info += row.sum() / total * self.calculate_entropy(row)

        # SplitInfo_A(D) = - sigma(j=1 to v) (D_j / D) * log(D_j / D, 2)
        sum_split = 0
        for row in df.values:
            sum_split += row.sum() / total * math.log(row.sum() / total, 2)

        sum_split = -sum_split

        # 실제 gain ratio 는 아님. 아래 값이 작으면 gain ratio가 큼.
        gain_ratio = sum_info / sum_split

        return gain_ratio

    # Maximum gain ratio 를 찾아 attribute 리턴.
    def maximum_gain_ratio(self):
        gain_ratio_dict = {}

        # attribute 하나 정해서 gain ratio 계산.
        attributes = self.data.columns[:-1]
        for attribute in attributes:
            gain_ratio_dict[attribute] = self.calculate_gain_ratio(attribute)

        return min(gain_ratio_dict.keys(), key=(lambda k: gain_ratio_dict[k]))

    # partitioning 프로세스를 stop 해야하면(다음 노드가 leaf 이면) True 리턴.
    def stop_partitioning(self, samples: pd.DataFrame):
        stop = False
        class_name = self.data.columns[-1]

        # 쪼갠 데이터의 클래스가 모두 같은 경우.
        if samples[class_name].nunique() == 1:
            stop = True

        # 더 이상 쪼갤 attribute 가 없을 경우.
        elif self.data.columns.nunique() == 1:
            stop = True

        # 샘플이 없는 경우.
        elif samples.empty:
            stop = True

        return stop

    # attribute 를 골라 트리를 분할.
    def expand(self):
        self.attribute = self.maximum_gain_ratio()
        labels = self.data[self.attribute].unique()

        for label in labels:
            # 정한 attribute 라벨이 label 인 튜플 필터링.
            samples = self.data[self.data[self.attribute] == label]

            # 쪼갠 데이터가 Conditions for stopping the partitioning process 에 걸리는지 확인.
            # 멈춰야하면 leaf
            if self.stop_partitioning(samples):
                self.children[label] = Node(samples, True)

            # 더 쪼갤 수 있으면 사용한 attribute 컬럼 제거한 데이터 내려보냄.
            else:
                self.children[label] = Node(samples.drop(self.attribute, axis=1))

    # 테스트 데이터 classify
    # iterrows 사용하면 pd.Series 오브젝트 리턴 됨.
    def test(self, test_data: pd.Series):
        # leaf 면 클래스 들고 나감.
        if self.is_leaf:
            return self.class_label

        # leaf 아니면 하위 노드로 재귀 탐색
        else:
            # 트레이닝 데이터에 없던 루트로 가면 자식노드가 없어서 에러남.
            try:
                label = test_data[self.attribute]
                return self.children[label].test(test_data)

            except KeyError:
                return self.class_label

    """ 
    지니인덱스
    def calculate_gini(self, df):
        df_part = df
        gini = 1

        labels = df_part.iloc[:, -1].unique()
        for label in labels:
            p = len(df_part[df_part.iloc[:, -1].isin([label])]) / len(df_part)
            gini -= p**2

        return gini

    # attribute 들의 gini index 를 계산 후 가장 작은(좋은) attribute 이름과 스플릿포인트 리턴.
    def gini_index(self):
        # 어떤 attribute 로 나눌지 gini 를 모두 계산해보자.
        attributes = self.data.columns

        # 각 attribute 의 지니 계수를 저장. {attribute: gini_index}
        gini_dict = {}

        # attribute 설정.
        for attribute in attributes:
            # 해당 attribute 에 어떤 값이 들어오는지 확보.
            s = set(self.data[attribute].unique())

            # attribute 를 기준으로 binary partition 하여 gini 계산.
            k = len(s)
            u = (k / 2) + 1
            gini_A_dict = {}
            while k >= u:
                for comb in combinations(attributes, k-1):
                    complement = s - set(comb)

                    # 특정 attribute 를 개수별로 반으로 가른 후 양쪽 확률 계산. 작은게 좋은 것.
                    D = self.data
                    D1 = self.data[self.data[attribute].isin(comb)]
                    D2 = self.data[self.data[attribute].isin(complement)]
                    gini_D1 = self.calculate_gini(self, D1)
                    gini_D2 = self.calculate_gini(self, D2)

                    gini_A = (len(D1) * gini_D1 + len(D2) * gini_D2) / len(D)

                    gini_A_dict[comb] = gini_A

            # 가장 작은 gini 를 가지는 스플릿 포인트
            split_point = min(gini_A_dict.keys(), key=(lambda i: gini_A_dict[i]))
            # attribute 의 최소 지니 계수.
            minimum_gini_A = gini_A_dict[split_point]
            # attribute 들끼리 비교 위해 딕셔너리에 추가.
            gini_dict[attribute] = {split_point: minimum_gini_A}

        # 각 attribute 의 지니 계수를 구한 후...
        best_attribute = min(gini_dict.keys(), key=(lambda i: gini_dict[i][1]))

        # attribute 이름과 해당 attribute를 골랐을 때 스플릿 포인트를 리턴.
        return {best_attribute: gini_dict[best_attribute][0]}
    """
