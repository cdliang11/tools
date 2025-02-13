
import argparse

from e2e_wfst.dict.vocab import Vocab

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract vocab')
    parser.add_argument('--slot_config', default=None, help='slot config file')
    parser.add_argument('lm', help='lm file')
    parser.add_argument('dir', help='output dir')
    args = parser.parse_args()
    vocab = Vocab(args.lm, args.slot_config)
    vocab.write_vocabs(args.dir)
