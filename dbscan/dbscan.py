# -*- coding: utf-8 -*-

# 2021 HYU. CSE
# Taehun Kim <th6424@gmail.com>

import numpy as np

UNVISITED = -2
OUTLIER = -1

class DBSCAN:
    def __init__(self, eps, min_pts):
        self.eps = eps
        self.min_pts = min_pts

        self.data = None
        self.labels = None
        self.distance_matrix = None

    def distance(self, p, q):
        return np.sum((p - q) ** 2) ** (1/2)

    def _make_distance_matrix(self):
        distance_matrix = np.zeros((self.data.shape[0], self.data.shape[0]), dtype=np.float32)
        for i in range(0, self.data.shape[0]):
            for j in range(i, self.data.shape[0]):
                distance_matrix[i][j] = distance_matrix[j][i] = np.sum((self.data[i] - self.data[j]) ** 2)

        return distance_matrix ** (1/2)

    def is_neighborhood(self, p, q):
        return self.distance_matrix[p][q] < self.eps

    # 모든 점에 대하여 타겟 포인트와 이웃인 것 찾기. 자신도 포함.
    def _find_neighbors(self, point_id):
        neighbors = []
        for i in range(0, self.data.shape[0]):
            if self.is_neighborhood(point_id, i):
                neighbors.append(i)

        return neighbors

    def _make_cluster(self, point_id, cluster_id):
        # 타겟 포인트가 코어포인트가 아니면 일단 아웃라이어.
        neighbors = self._find_neighbors(point_id)
        if len(neighbors) < self.min_pts:
            self.labels[point_id] = OUTLIER

            return False

        # 고른게 코어면 클러스터 생성.
        else:
            print(f'{cluster_id}번 클러스터 생성 중')

            self.labels[point_id] = cluster_id
            # 코어 주변 이웃들 클러스터에 편입.
            for neighbor_id in neighbors:
                self.labels[neighbor_id] = cluster_id

            # 클러스터 확장 여부 판단.
            while len(neighbors) > 0:
                new_point = neighbors[0]
                new_neighbors = self._find_neighbors(new_point)  # _find_neighbors로 받아온 이웃 리스트는 이웃의 id로 이루어짐.

                # 이웃 포인트에 포커스, 코어 포인트면...
                if len(new_neighbors) >= self.min_pts:
                    for new_neighbor in new_neighbors:
                        # 새 포인트의 이웃이 미방문: 클러스터에 편입하며 코어 검사 (reachable 쭉 이어감).
                        # 아웃라이어: 이미 방문했는데 코어가 아니었던 포인트.
                        if self.labels[new_neighbor] == UNVISITED or self.labels[new_neighbor] == OUTLIER:
                            if self.labels[new_neighbor] == UNVISITED:
                                neighbors.append(new_neighbor)

                            self.labels[new_neighbor] = cluster_id

                neighbors = neighbors[1:]

            return True

    def get_labels(self):
        return self.labels

    def run(self, data: np.ndarray):
        # id 제거
        print('데이터 가공')
        self.data = data[:, 1:].astype(np.float32)

        # 오브젝트간 거리 행렬 연산.
        print('거리 연산')
        self.distance_matrix = self._make_distance_matrix()

        # 모든 점의 라벨 UNVISITED로 초기화.
        print('라벨 초기화')
        self.labels = np.full(self.data.shape[0], UNVISITED)

        cluster_id = 0
        # 모든 점을 순회
        for i in range(0, self.data.shape[0]):
            if self.labels[i] == UNVISITED:
                # 고른 포인트가 코어포인트라서 클러스터를 만들면 작업 후 다음 클러스터 아이디 증가.
                if self._make_cluster(i, cluster_id):
                    cluster_id = cluster_id + 1

        return self.get_labels()
