import random

jobs = [
    [(0, 3), (1, 2), (2, 2)],
    [(0, 2), (2, 1), (1, 4)],
    [(1, 4), (2, 3), (0, 2)],
    [(2, 2), (0, 3), (1, 2)]
]

N = 4
POP = 30
GEN = 50

def create():
    x = [i for i in range(N) for _ in jobs[i]]
    random.shuffle(x)
    return x

def makespan(x):
    machine = [0] * 3
    jobtime = [0] * N
    op = [0] * N

    for j in x:
        k = op[j]
        m, t = jobs[j][k]
        start = max(machine[m], jobtime[j])
        finish = start + t
        machine[m] = finish
        jobtime[j] = finish
        op[j] += 1

    return max(jobtime)

def select(pop):
    return min(random.sample(pop, 3), key=makespan)

def crossover(a, b):
    p, q = sorted(random.sample(range(len(a)), 2))
    child = [-1] * len(a)
    child[p:q] = a[p:q]

    count = [0] * N
    for x in child:
        if x != -1:
            count[x] += 1

    pos = q

    for x in b:
        if count[x] < 3:
            while child[pos % len(child)] != -1:
                pos += 1
            child[pos % len(child)] = x
            count[x] += 1
            pos += 1

    return child

def mutate(x):
    if random.random() < 0.2:
        a, b = random.sample(range(len(x)), 2)
        x[a], x[b] = x[b], x[a]

pop = [create() for _ in range(POP)]

for _ in range(GEN):
    new = [min(pop, key=makespan)]

    while len(new) < POP:
        a = select(pop)
        b = select(pop)
        c = crossover(a, b)
        mutate(c)
        new.append(c)

    pop = new

best = min(pop, key=makespan)

print("Best Schedule:", [x + 1 for x in best])
print("Minimum Makespan:", makespan(best))
