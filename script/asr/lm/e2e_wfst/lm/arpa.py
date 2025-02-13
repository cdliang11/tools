#!/usr/bin/env python3

class Arpa:

    def __init__(self, arpa_file):
        self.vocab = self.read_arpa(arpa_file)

    def read_arpa(self, arpa_file):
        vocab = set()
        with open(arpa_file) as fin:
            # ignore until "\1-grams:"
            contains_1gram = False
            while True:
                line = fin.readline()
                if not line:
                    break
                if line.strip() == '\\1-grams:':
                    contains_1gram = True
                    break
            assert contains_1gram
            while True:
                line = fin.readline().strip()
                if not line:
                    break
                if line == '\\2-grams:' or line == '\\end\\':
                    break
                if line == '':
                    continue
                arr = line.split()
                assert len(arr) >= 2
                # $x is kept for class lm
                if not arr[1].startswith('$'):
                    vocab.add(arr[1])
        return sorted(vocab)


if __name__ == '__main__':
    import sys
    for w in Arpa(sys.argv[1]).vocab:
        print(w)
