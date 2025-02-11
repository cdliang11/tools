
import os
import re


class Lexicon:

    def __init__(self):
        self.lexicons = {}

    def vocabs(self):
        return sorted(self.lexicons.keys())

    def read_lexicon(self, lexicon_file):
        with open(lexicon_file, 'r', encoding='utf8') as fin:
            for line in fin:
                arr = line.strip().split()
                w = arr[0]
                if w not in self.lexicons:
                    self.lexicons[w] = []
                self.lexicons[w].append(arr[1:])

    def read_e2e_unit(self, e2e_unit_file):
        unit_table = set()
        with open(e2e_unit_file, 'r', encoding='utf8') as fin:
            for line in fin:
                unit = line.split()[0]
                unit_table.add(unit)
        return unit_table

    def read_fix_dict(self, unit_set, fix_dict_file=None):
        fix_dict = {}
        if fix_dict_file is not None:
            assert os.path.exists(fix_dict_file)
            with open(fix_dict_file, 'r', encoding='utf8') as fin:
                for line in fin:
                    arr = line.strip().split()
                    word = arr[0]
                    prons = arr[1:]
                    for u in prons:
                        assert u in unit_set
                    if word not in fix_dict:
                        fix_dict[word] = []
                    fix_dict[word].append(prons)
        return fix_dict

    def contain_oov(self, pron, unit_set):
        for u in pron:
            if u not in unit_set:
                return True
        return False

    def parse_from_files(self,
                         e2e_unit_file,
                         vocab_file,
                         bpe_model_file=None,
                         fix_dict_file=None):
        """ Generate lexicon by e2e_unit_file, bpe_model_file, fix_dict_file
            for each word in vocab_file. If the word is in fix_dict_file,
            just use the prounication in fix_dict_file, otherwise generate
            prounication by e2e_unit_file and bpe_model_file
        Args:
            e2e_unit_file: model unit file of e2e model,
                each line in format: <token> <id>
            vocab_file: vocab file for LM, each line in format: <word>
            bpe_model_file: optional, bpe model file for english words
            fix_dict_file: optional, lexicon for fix OOV or bugs,
                each line in format: <word> <unit1> <unit2> <...>
        """
        unit_set = self.read_e2e_unit(e2e_unit_file)
        fix_dict = self.read_fix_dict(unit_set, fix_dict_file)
        bpemode = bpe_model_file is not None
        if bpemode:
            assert os.path.exists(bpe_model_file)
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor()
            sp.Load(bpe_model_file)
        with open(vocab_file, 'r', encoding='utf8') as fin:
            for line in fin:
                word = line.strip().split()[0]
                if word == '<SPOKEN_NOISE>':
                    continue
                else:
                    if word in self.lexicons:
                        continue
                    if word in fix_dict:
                        self.lexicons[word] = fix_dict[word]
                    else:
                        if re.match('[0-9]', word):
                            print('Ignoring word {} which contains digit'.
                                  format(word))
                            continue

                        pron = []
                        pattern = re.compile(r'([\u4e00-\u9fff]+)')
                        word_segments = pattern.split(word)
                        for segment in word_segments:
                            if re.match('^[a-zA-Z\']+$', segment):
                                if bpemode:
                                    en_pron = sp.EncodeAsPieces(segment.upper())
                                else:
                                    # Optional, append ▁ in front of english word
                                    en_pron = '▁' + segment.upper()
                                pron.extend(en_pron)
                            else:
                                # UTF-8 Chinese word segment
                                for utf8_unit in segment:
                                    pron.append(utf8_unit)

                        if self.contain_oov(pron, unit_set):
                            print('Ignoring word {} which contains OOV unit'.
                                  format(word))
                            continue
                        self.lexicons[word] = [pron]

    def write_lexicon(self, lexicon_file):
        vocabs = sorted(self.lexicons.keys())
        with open(lexicon_file, 'w', encoding='utf8') as fout:
            for w in vocabs:
                prons = self.lexicons[w]
                for pron in prons:
                    fout.write('{} {}\n'.format(w, ' '.join(pron)))

    def write_lexicon_fst(self, lexicon_fst_file):
        vocabs = sorted(self.lexicons.keys())

        loop_state = 0
        next_state = 1
        with open(lexicon_fst_file, 'w', encoding='utf8') as fout:
            for w in vocabs:
                prons = self.lexicons[w]
                for pron in prons:
                    s = loop_state
                    word_or_eps = w
                    for i, u in enumerate(pron):
                        if i < len(pron) - 1:
                            ns = next_state
                            next_state += 1
                        else:
                            ns = loop_state
                        fout.write('{}\t{}\t{}\t{}\n'.format(
                            s, ns, u, word_or_eps))
                        word_or_eps = "<eps>"
                        s = ns
            fout.write('{}\t0\n'.format(loop_state))
