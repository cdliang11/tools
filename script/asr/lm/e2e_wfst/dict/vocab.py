
import os

from e2e_wfst.lm.arpa import Arpa
from e2e_wfst.fst.slot import Slots


# Vocab write the following vocabs:
# 1. lm.vocab: vocab of LM
# 2. class.vocab: vocab of class slot lists
# 3. lexicon.vocab: union(lm.vocab, class.vocab)
class Vocab:

    def __init__(self,
                 lm_file,
                 slot_config_file=None):
        lm_vocab = set(Arpa(lm_file).vocab)
        self.lm_vocab = sorted(lm_vocab)
        self.class_vocab = []
        if slot_config_file is not None:
            self.class_vocab = Slots(slot_config_file).vocab
        self.lexicon_vocab = set(self.lm_vocab + self.class_vocab)

    # Write Vocabs
    def write_vocab(self, vocab, vocab_file):
        vocab = sorted(vocab)
        with open(vocab_file, 'w', encoding='utf8') as fout:
            for w in vocab:
                fout.write('{}\n'.format(w))

    def write_vocabs(self, dir):
        self.write_vocab(self.lm_vocab, os.path.join(dir, 'lm.vocab'))
        self.write_vocab(self.class_vocab, os.path.join(dir, 'class.vocab'))
        self.write_vocab(self.lexicon_vocab,
                         os.path.join(dir, 'lexicon.vocab'))
