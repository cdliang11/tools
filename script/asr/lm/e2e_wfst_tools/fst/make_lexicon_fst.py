#!/usr/bin/env python

import argparse

from e2e.dict.lexicon import Lexicon

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='make lexicon fst')
    parser.add_argument('lexicon', help='input lexicon file')
    parser.add_argument('lexicon_fst', help='output lexicon txt file')
    args = parser.parse_args()

    lexicon = Lexicon()
    lexicon.read_lexicon(args.lexicon)
    lexicon.write_lexicon_fst(args.lexicon_fst)
