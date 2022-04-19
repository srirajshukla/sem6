def emptySet():
    return []

def isEmpty(S):
    return len(S)==0

def isMember(S, e):
    for s in S:
        if s==e:
            return True

    return False

def singleton(x):
    return [x]

def isSubset(P, Q):
    for p in P:
        if not isMember(Q, p):
            return False
    return True

def setEqual(P, Q):
    return isSubset(P, Q) and isSubset(Q, P)

def union(P, Q):
    U = P[:]
    for q in Q:
        if not isMember(U, q):
            U.append(q)
    return U

def intersection(P, Q):
    I = []
    for p in P:
        if isMember(Q, p):
            I.append(p)
    return I

def cartesian(P, Q):
    c = []
    for p in P:
        for q in Q:
            c.append((p, q))
    return c

def power(P):
    powerSet = [[]]

    for p in P:
        tmp = []
        for s in powerSet:
            nv = s[:]
            nv.append(p)
            tmp.append(nv)
        powerSet.extend(tmp)
    return powerSet


def emptyset_2():
    return []

def isempty_2(S):
    return len(S)==0

def member_2(S, e):
    """
    S is a sorted list
    check if e is present in S or not
    """

    i = 0
    j = len(S)-1
    while i <= j:
        m = i + (j-i)//2
        if S[m] == e:
            return True
        elif S[m] < e:
            i = m+1
        else:
            j = m-1
    return False

def singleton_2(x):
    return [x]

def isSubset_2(P, Q):
    """
    P is a sorted list
    Q is a sorted list
    """
    for p in P:
        if not member_2(Q, p):
            return False
    return True

def setEqual_2(P, Q):
    if len(P) != len(Q):
        return False
    l = len(P)
    for i in range(l):
        if P[i] != Q[i]:
            return False
    return True

def union_2(P, Q):
    """
    P is a sorted list
    Q is a sorted list
    """
    m = len(P)
    n = len(Q)
    i, j = 0, 0
    U = []
    while i < m and j < n:
        if P[i] < Q[j]:
            U.append(P[i])
            i += 1
        elif P[i] > Q[j]:
            U.append(Q[j])
            j += 1
        else:
            U.append(P[i])
            i += 1
            j += 1

    return U

def intersection_2(P, Q):
    """
    P is a sorted list
    Q is a sorted list
    """
    m = len(P)
    n = len(Q)
    i, j = 0, 0
    I = []
    while i < m and j < n:
        if P[i] < Q[j]:
            i += 1
        elif P[i] > Q[j]:
            j += 1
        else:
            I.append(P[i])
            i += 1
            j += 1

    return I

def cartesian_2(P, Q):
    """
    P is a sorted list
    Q is a sorted list
    """
    c = []
    for p in P:
        for q in Q:
            c.append((p, q))
    return c

def power_2(P):
    powerSet = [[]]

    for p in P:
        tmp = []
        for s in powerSet:
            nv = s[:]
            nv.append(p)
            tmp.append(nv)
        powerSet.extend(tmp)
    return powerSet

if __name__ == "__main__":
    print(emptySet())
    print(isMember([4, 2, 3, 1], 4))
    print(singleton(4))
    print(isSubset([1, 2, 3], [3, 2, 1, 4]))
    print(setEqual([1, 2, 3], [3, 2, 1]))
    print(union(["MTL100", "COL100"], ["NLN100", "MTL100"]))
    print(intersection(["MTL100", "COL100"], ["NLN100", "MTL100"]))
    print(cartesian([1, 3, 2], ["C", "M"]))
    print(power([1, 3, 2]))


