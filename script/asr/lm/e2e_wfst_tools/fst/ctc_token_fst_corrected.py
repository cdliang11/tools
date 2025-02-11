#!/usr/bin/env python

import sys

basic_units = []
disambig_units = []
with open(sys.argv[1], 'r', encoding='utf8') as fin:
    for line in fin:
        arr = line.split()
        unit = arr[0]
        if unit == '<eps>' or unit == '<blank>':
            continue
        elif unit.startswith('#') or unit.startswith('@'):
            disambig_units.append(unit)
        else:
            basic_units.append(unit)

# 1. add start state
print('0 0 <blank> <eps>')

# 2. 0 -> i, i -> i, i -> 0
for i, u in enumerate(basic_units):
    s = i + 1  # state
    print('{} {} {} {}'.format(0, s, u, u))
    print('{} {} {} {}'.format(s, s, u, '<eps>'))
    print('{} {} {} {}'.format(s, 0, '<blank>', '<eps>'))

# 3. i -> other unit
for i, u1 in enumerate(basic_units):
    s = i + 1
    for j, u2 in enumerate(basic_units):
        if i != j:
            print('{} {} {} {}'.format(s, j + 1, u2, u2))

# 4. add disambiguous arcs on every final state
for i in range(0, len(basic_units) + 1):
    for u in disambig_units:
        print('{} {} {} {}'.format(i, i, '<eps>', u))

# 5. every i is final state
for i in range(0, len(basic_units) + 1):
    print(i)
