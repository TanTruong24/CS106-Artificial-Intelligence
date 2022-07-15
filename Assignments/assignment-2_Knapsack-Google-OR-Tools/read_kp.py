
from sqlalchemy import values


def read_test():
    with open('./s000.kp', 'r') as f:
        testcase = f.read()

    lines = testcase.split('\n')

    capacities = [int(lines[2])]
    values = []
    weights = []
    temp_weights = []
    for i in range(4, len(lines)-1):
        lst = lines[i].split()
        values.append(int(lst[0]))
        temp_weights.append(int(lst[1]))

    weights.append(temp_weights)

    return values, weights, capacities


v, w, c = read_test()
print(v)
print(w)
print(c)
