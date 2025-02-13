#!/usr/bin/env python

import argparse

from e2e_wfst.fst.slot import Slots

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='prepare input and output symbols')
    parser.add_argument('slot_config', default=None, help='slot config file')
    parser.add_argument('symbols', default=None, help='symbol table file')
    parser.add_argument('ifst', help='input G.fst')
    parser.add_argument('ofst', help='output G.fst')
    args = parser.parse_args()

    slots = Slots(args.slot_config)

    slots.write_fst_and_replace(args.symbols, args.ifst, args.ofst)
