import numpy as np
from math import gcd
from scipy.special import comb
from itertools import combinations


def generatorPoints(N, M, method='NBI'):
    if method == 'NBI':
        (W, N) = NBI(N, M)
    elif method == 'Latin':
        (W, N) = Latin(N, M)
    elif method == 'MUD':
        (W, N) = MixtureUniformDesign(N, M)
    elif method == 'ILD':
        (W, N) = ILD(N, M)
    return (W, N)


def NBI(N, M):
    H1 = 1
    while comb(H1 + M,
               M - 1) <= N:  # comb(H1+M, M-1) Combination number (binomial coefficient) "select (M-1) numbers from (H1+M)"
        H1 = H1 + 1
    s = range(1, H1 + M)
    W = np.asarray(list(combinations(s, M - 1))) - np.tile(np.arange(0, M - 1), (int(comb(H1 + M - 1, M - 1)), 1)) - 1

    # W = np.array(W) - np.tile(np.arange(0, M-1), (int(comb(H1+M-1, M-1)), 1)) - 1
    W = (np.append(W, np.zeros((W.shape[0], 1)) + H1, axis=1) - np.append(np.zeros((W.shape[0], 1)), W, axis=1)) / H1
    if H1 < M:
        H2 = 0
        while comb(H1 + M - 1, M - 1) + comb(H2 + M, M - 1) <= N:
            H2 += 1
        if H2 > 0:
            W2 = []
            s2 = range(1, H2 + M)
            W2 = np.asarray(list(combinations(s2, M - 1))) - np.tile(np.arange(0, M - 1),
                                                                     (int(comb(H2 + M - 1, M - 1)), 1)) - 1
            # W2 = np.array(W2) - np.tile(np.arange(0, M-1), (int(comb(H2+M-1, M-1)), 1)) - 1
            W2 = (np.append(W2, np.zeros((W2.shape[0], 1)) + H2, axis=1) - np.append(np.zeros((W2.shape[0], 1)), W2,
                                                                                     axis=1)) / H2
            W = np.append(W, W2 / 2 + 1 / (2 * M), axis=0)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            if W[i, j] < 1e-6:
                W[i, j] = 1e-6
    N = W.shape[0]
    return (W, N)


def Latin(N, M):
    W = np.random.random((N, M))
    W = np.argsort(W, axis=0, kind='mergesort') + 1
    W = (np.random.random((N, M)) + W - 1) / N
    return (W, N)


def ILD(N, M):
    In = M * np.eye(M)
    W = np.zeros((1, M))
    edgeW = W
    while np.shape(W)[0] < N:
        edgeW = np.tile(edgeW, (M, 1)) + np.repeat(In, np.shape(edgeW)[0], axis=0)
        edgeW = np.unique(edgeW, axis=0)
        ind = np.where(np.min(edgeW, axis=0) == 0)[0]
        edgeW = np.take(edgeW, ind, axis=0)
        W = np.append(W + 1, edgeW, axis=0)
    W = W / np.tile(np.sum(W, axis=1)[:, np.newaxis], (np.shape(W)[1],))
    W = np.where(W > 1e6, 1e6, W)
    N = np.shape(W)[0]
    return W, N


def MixtureUniformDesign(N, M):
    X = GoodLatticePoint(N, M - 1) ** (1 / np.tile(np.arange(M - 1, 0, -1), (N, 1)))
    X = np.clip(X, -np.infty, 1e6)
    X = np.where(X == 0, 1e-12, X)
    W = np.zeros((N, M))
    W[:, :-1] = (1 - X) * np.cumprod(X, axis=1) / X
    W[:, -1] = np.prod(X, axis=1)
    return W, N


def GoodLatticePoint(N, M):
    range_nums = np.arange(1, N + 1, 1)
    ind = np.asarray([], dtype=np.int64)
    for i in range(np.size(range_nums)):
        if gcd(range_nums[i], N) == 1:
            ind = np.append(ind, i)
    W1 = range_nums[ind]
    W = np.mod(np.dot(np.arange(1, N + 1, 1).reshape(-1, 1), W1.reshape(1, -1)), N)
    W = np.where(W == 0, N, W)
    nCombination = int(comb(np.size(W1), M))
    if nCombination < 1e4:
        Combination = np.asarray(list(combinations(np.arange(1, np.size(W1) + 1, 1), M)))
        CD2 = np.zeros((nCombination, 1))
        for i in range(nCombination):
            tmp = Combination[i, :].tolist()
            UT = np.empty((np.shape(W)[0], len(tmp)))
            for j in range(len(tmp)):
                UT[:, j] = W[:, tmp[j] - 1]
            CD2[i] = CalCD2(UT)
        minIndex = np.argmin(CD2)
        tmp = Combination[minIndex, :].tolist()
        Data = np.empty((np.shape(W)[0], len(tmp)))
        for j in range(len(tmp)):
            Data[:, j] = W[:, tmp[j] - 1]
    else:
        CD2 = np.zeros((N, 1))
        for i in range(N):
            UT = np.mod(np.dot(np.arange(1, N + 1, 1).reshape(-1, 1), (i + 1) ** np.arange(0, M, 1).reshape(1, -1)), N)
            CD2[i] = CalCD2(UT)
        minIndex = np.argmin(CD2)
        Data = np.mod(
            np.dot(np.arange(1, N + 1, 1).reshape(-1, 1), (minIndex + 1) ** np.arange(0, M, 1).reshape(1, -1)), N)
        Data = np.where(Data == 0, N, Data)
    Data = (Data - 1) / (N - 1)
    return Data


def CalCD2(UT):
    N, S = np.shape(UT)
    X = (2 * UT - 1) / (2 * N)
    CS1 = np.sum(np.prod(2 + np.abs(X - 1 / 2) - (X - 1 / 2) ** 2, axis=1))
    CS2 = np.zeros((N, 1))
    for i in range(N):
        CS2[i] = np.sum(np.prod((1 + 1 / 2 * np.abs(np.tile(X[i, :], (N, 1)) - 1 / 2)
                                 + 1 / 2 * np.abs(X - 1 / 2)
                                 - 1 / 2 * np.abs(np.tile(X[i, :], (N, 1)) - X)), axis=1))
    CS2 = np.sum(CS2)
    CD2 = (13 / 12) ** S - 2 ** (1 - S) / N * CS1 + 1 / (N ** 2) * CS2
    return CD2


class splitWeights():
    def __init__(self, vector) -> None:
        self.vector = vector
        self.nums = 0


def InitializeVector(M):
    # W, Nw = generatorPoints(self.H2, M)
    W, Nw = generatorPoints(5, M)
    d = 0.7
    W = W * d + (1 - d) / M
    W0, _ = generatorPoints(1, M)
    Nw += 1
    W = np.append(W, W0, axis=0)
    Vs = [splitWeights(W[i, :]) for i in range(Nw)]
    return Vs


def identificationCurvature(PObjs, vectors, K=None, numsOfNeighbor=np.infty):
    N = np.shape(PObjs)[0]
    E = getExtremePoints(PObjs)

    uPObjs = PObjs
    # normalization
    # uPObjs = (PObjs - np.tile(E[0, :], (N, 1))) / np.tile(E[1, :] - E[0, :], (N, 1))
    Ri, _ = AssociationWeights(uPObjs, vectors)
    Ri = Ri.astype(int)
    w_size = np.shape(vectors)[0]

    ifKnees = np.zeros(N)
    for j in range(w_size):
        wj_index = np.where(Ri == j)[0]
        if np.size(wj_index) == 0:
            continue
        curvatures = np.ones(np.size(wj_index)) * np.infty
        curvatures = calcCurvature(uPObjs[wj_index, :])
        if K is None:
            soi = np.where(curvatures == np.min(curvatures))[0]  # Solution of convex region corresponding to minimum curvature
            ifKnees[wj_index[soi]] = 1
        elif K == w_size:
            soi = np.where(curvatures == np.min(curvatures))[
                0]  # Solution of convex region corresponding to minimum curvature
            ifKnees[wj_index[soi]] = 1
        else:
            numsOfSoi = 0
            for i in range(len(wj_index)):
                dis = calculateDistMatrix(uPObjs[[wj_index[i]], :], uPObjs)
                ind = np.sort(dis)
                nl = int(min(numsOfNeighbor, len(wj_index) + 1, len(ind)))
                if nl == 1:
                    ifKnees[wj_index[i]] = 1
                    numsOfSoi += 1
                    continue
                neighbors = np.take(curvatures, ind[1:nl], 0)
                if np.min(neighbors) == curvatures[i]:  # If its curvature is the smallest in the field, it may be a promising solution
                    ifKnees[wj_index[i]] = 1
                    numsOfSoi += 1
                    if numsOfSoi == K:
                        break
    ind = np.where(ifKnees == 1)[0]
    knee_points = PObjs[ind, :]
    return ind, knee_points


def getExtremePoints(Objs, transpose=False):
    N, M = np.shape(Objs)
    E = np.zeros((2, M))
    # tmp1 -- ideal point
    # tmp2 -- nadir point
    for m in range(M):
        tmp1 = np.inf
        tmp2 = -np.inf
        for i in range(N):
            if tmp1 > Objs[i, m]:
                tmp1 = Objs[i, m]
            elif tmp2 < Objs[i, m]:
                tmp2 = Objs[i, m]
        E[0, m] = tmp1
        E[1, m] = tmp2
    if transpose:
        extremes = np.zeros((2, M))
        for i in range(M):
            extremes[i, :] = E[0, :]
            extremes[i, i] = E[1, i]
        return extremes
    return E


def AssociationWeights(PopObjs, W):
    N, M = np.shape(PopObjs)
    Nid = np.zeros(M)
    W_size = np.shape(W)[0]

    Ri = np.zeros(N, dtype=int)
    Rc = np.zeros(W_size, dtype=int)

    for i in range(N):
        dis = np.zeros(W_size)
        for j in range(W_size):
            d, sums = 0, np.linalg.norm(W[j, :], ord=2)
            for k in range(M):
                d += np.abs((PopObjs[i, k] - Nid[k]) * W[j, k] / sums)
            d2 = 0
            for k in range(M):
                d2 += (PopObjs[i, k] - (Nid[k] + d * W[j, k])) ** 2
            dis[j] = np.sqrt(d2)
        Index = np.where(dis == min(dis))
        index = Index[0][0]
        Ri[i] = index
        Rc[index] += 1
    return Ri, Rc


def calcCurvature(uPObjs):
    uPObjs = np.where(uPObjs > 1e-12, uPObjs, 1e-12)
    N, M = np.shape(uPObjs)
    P = np.ones(N)  # Initial curvature
    lamda = 1 + np.zeros(N)
    E = np.sum(uPObjs ** np.tile(P[:, np.newaxis], (M,)), axis=1) - 1
    for epoch in range(5000):
        # gradient descent
        G = np.sum(uPObjs ** np.tile(P[:, np.newaxis], (M,)) * np.log(uPObjs), axis=1)
        newP = P - lamda * E * G
        newE = np.sum(uPObjs ** np.tile(newP[:, np.newaxis], (M,)), axis=1) - 1
        # print("newE:{}".format(newE))
        # Update the value of each weight
        update = (newP > 0) & (np.sum(newE ** 2) < np.sum(E ** 2))
        # print("update:{}".format(update))
        P[update] = newP[update]
        E[update] = newE[update]
        lamda[update] = lamda[update] * 1.1
        lamda[~update] = lamda[~update] / 1.1
    return P


def calculateDistMatrix(datas, DATAS):
    dist = np.zeros((datas.shape[0], DATAS.shape[0]))  # the distance matrix
    if datas.shape[1] > 1:
        for i in range(datas.shape[0]):
            Temp = np.sum((DATAS - np.dot(np.ones((DATAS.shape[0], 1)), datas[i, :][np.newaxis, :])) ** 2, axis=1)
            dist[i, :] = np.sqrt(Temp)
    else:  # 1-D data
        for i in range(datas.shape[0]):
            dist[i, :] = np.abs(datas[i] - DATAS)
    return dist


