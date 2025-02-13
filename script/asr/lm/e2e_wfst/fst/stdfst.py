#!/usr/bin/env python

import sys

# Python code for openfst VectorFst<StdArc> implementation.
# The weight in StdArc is Tropical weight.
# Tropical semiring: (min, +, inf, 0)


class Arc:

    def __init__(self, ilabel, olabel, weight, nextstate):
        self.ilabel = ilabel
        self.olabel = olabel
        self.weight = weight
        self.nextstate = nextstate


class State:

    def __init__(self):
        self.arcs = []
        # Zero value in tropical weight
        self.weight = float('inf')


class Fst:

    def __init__(self, txt_fst_file=None):
        self.states = []
        self.start = -1  # kNoStateId
        self.finals = set()
        if txt_fst_file is not None:
            self.read_txt_fst(txt_fst_file)

    def add_state(self):
        self.states.append(State())
        return len(self.states) - 1

    def set_start(self, s):
        self.start = s

    def set_final(self, s, weight=0.0):
        self.expand_state(s)
        self.finals.add(s)
        self.states[s].weight = weight

    def add_arc(self, s, arc):
        self.expand_state(s)
        self.states[s].arcs.append(arc)

    # Expand maximum state to s
    def expand_state(self, s):
        while len(self.states) < s + 1:
            self.add_state()

    def read_txt_fst(self, txt_fst_file):
        self.states = []
        self.start = -1  # kNoStateId
        self.finals = set()
        """ Read fst from txt fst file
        arc format: src dest ilabel olabel [weight]
        final state format: state [weight]
        lines may occur in any order except initial state must be first line
        unspecified weights default to 0.0(for the library-default Weight type)
        """
        with open(txt_fst_file, 'r', encoding='utf8') as fin:
            first = True
            for line in fin:
                arr = line.strip().split()
                if len(arr) == 0:
                    continue
                src = int(arr[0])
                if first:
                    self.set_start(src)
                    first = False
                if len(arr) in [1, 2]:
                    weight = 0.0 if len(arr) == 1 else float(arr[0])
                    self.set_final(src, weight)
                elif len(arr) in [4, 5]:
                    dst = int(arr[1])
                    ilabel = arr[2]
                    olabel = arr[3]
                    weight = 0.0 if len(arr) == 4 else float(arr[4])
                    self.add_arc(src, Arc(ilabel, olabel, weight, dst))
                else:
                    print('Unexpected line {}'.format(line))
                    sys.exit(1)

    def _write_state_info(self, s, fout):
        state = self.states[s]
        for arc in state.arcs:
            fout.write('{} {} {} {} {}\n'.format(s, arc.nextstate, arc.ilabel,
                                                 arc.olabel, arc.weight))
        if s in self.finals:
            # Ignore the weight here
            fout.write('{}\n'.format(s))

    def write_txt_fst(self, txt_fst_file):
        with open(txt_fst_file, 'w', encoding='utf8') as fout:
            self._write_state_info(self.start, fout)
            for s, state in enumerate(self.states):
                if s != self.start:
                    self._write_state_info(s, fout)


if __name__ == '__main__':
    fst = Fst()
    fst.read_txt_fst(sys.argv[1])
    fst.write_txt_fst(sys.argv[2])
