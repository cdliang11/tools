
import argparse

from e2e_wfst.dict.lexicon import Lexicon

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='prepare dict')
    parser.add_argument('--bpe_model', default=None, help='bpe model file')
    parser.add_argument('--fix_dict', default=None, help='fix lexicon file')
    parser.add_argument('e2e_unit', help='input e2e unit file')
    parser.add_argument('vocab', help='input vocab file')
    parser.add_argument('lexicon', help='output lexicon file')
    args = parser.parse_args()

    lexicon = Lexicon()
    lexicon.parse_from_files(args.e2e_unit, args.vocab, args.bpe_model,
                             args.fix_dict)

    lexicon.write_lexicon(args.lexicon)
