#!/usr/bin/env python

import argparse

from e2e_wfst.dict.lexicon import Lexicon
from e2e_wfst.fst.slot import Slots


def read_e2e_unit(e2e_unit_file):
    e2e_units = []
    with open(e2e_unit_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            e2e_units.append(arr[0])
    return e2e_units


def write_symbols(symbols, file_name):
    with open(file_name, 'w', encoding='utf8') as fout:
        for i, symbol in enumerate(symbols):
            fout.write('{}\t{}\n'.format(symbol, i))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='prepare input and output symbols')
    parser.add_argument('--slot_config', default=None, help='slot config file')
    parser.add_argument('e2e_unit', help='input e2e unit file')
    parser.add_argument('lexicon', help='lexicon file')
    parser.add_argument('ndisambig',
                        type=int,
                        help='number of disambig sysmbols in lexicon')
    parser.add_argument('isymbols', help='isymbols of the final graph')
    parser.add_argument('osymbols', help='osymbols of the final graph')
    args = parser.parse_args()

    e2e_units = read_e2e_unit(args.e2e_unit)

    lexicon = Lexicon()
    lexicon.read_lexicon(args.lexicon)

    slots = Slots(args.slot_config)
    """
    Input symbols of the final graph
    <eps> 0     # <eps> in FST
    u1          # e2e unit
    u2
    ...
    @1          # @1,@2, lexicon disambigous symbol
    @2
    ...
    #0          # #0, LM backoff auxiliary symbol in G
    #1          # #1,#2, ..., class auxiliary symbol in G
    #2
    ...
    """
    isymbols = ['<eps>']
    isymbols.extend(e2e_units)
    for i in range(1, args.ndisambig + 1):
        isymbols.append('@{}'.format(i))
    isymbols.append('#0')
    isymbols.extend(slots.disambigs())
    write_symbols(isymbols, args.isymbols)
    """
    Output symbols of the final graph
    <eps> 0     # <eps> in FST
    $root       # placeholder reserved for root FST
    $class1     # placeholder for further replace in the following pipeline
    $class2
    ...
    <class1>    # class tag in
    <class2>
    ...
    w1
    w2
    ...
    #0          # #0, LM backoff auxiliary symbol in G
    #1          # #1,#2, ..., class auxiliary symbol in G
    #2
    ...
    """
    osymbols = ['<eps>', '$root']
    osymbols.extend(slots.replaces())
    osymbols.extend(slots.tags())
    osymbols.extend(lexicon.vocabs())
    osymbols.append('#0')
    osymbols.extend(slots.disambigs())
    write_symbols(osymbols, args.osymbols)
