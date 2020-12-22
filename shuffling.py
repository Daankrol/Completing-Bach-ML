import random


def getperm(l):
    # seed = sum(l)
    seed = 42
    random.seed(seed)   #
    perm = list(range(len(l)))
    random.shuffle(perm)
    random.seed()  # optional, in order to not impact other code based on random
    return perm


def shuffle(l):  # [1, 2, 3, 4]
    perm = getperm(l)  # [3, 2, 1, 0]
    l[:] = [l[j] for j in perm]  # [4, 3, 2, 1]


def unshuffle(l):  # [4, 3, 2, 1]
    perm = getperm(l)  # [3, 2, 1, 0]
    res = [None] * len(l)  # [None, None, None, None]
    for i, j in enumerate(perm):
        res[j] = l[i]
    l[:] = res  # [1, 2, 3, 4]
