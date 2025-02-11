#!/usr/bin/env python

import copy
import math
import os
from e2e.fst.stdfst import Arc, Fst
"""
There are three cases in total for slot config:
1. {}: emtpy slot, which is kept for replace on the fly for personalized
       or on-device recognition.
2. txt fst file: which is ending with .fst.txt, the slot is prebuilt and
       convert to fst in text format. Typically, it's built by Thrax for
       cases like phone number, flight number and so on.
3. entity_file: which is a common file, in which every line is one case of
       this entity.
"""


class Slot:

    def __init__(self, index, name, config):
        self.index = index
        self.name = name
        self.tag = '<{}>'.format(self.name)
        self.replace = '${}'.format(self.name)
        self.disambig = '#{}'.format(self.index - 1)
        self.vocab = []
        self.is_empty = False
        self.is_fst = False

    def write_slot_fst(self, fst_file):
        pass


class EmptySlot(Slot):

    def __init__(self, index, name, config):
        super().__init__(index, name, config)
        self.is_empty = True


class FstSlot(Slot):

    def __init__(self, index, name, config):
        super().__init__(index, name, config)
        self.is_fst = True
        self.ifst = Fst(config)
        vocab = set()
        for s in self.ifst.states:
            for arc in s.arcs:
                vocab.add(arc.ilabel)
                vocab.add(arc.olabel)
        self.vocab = sorted(vocab)

    def write_slot_fst(self, fst_file):
        ofst = copy.deepcopy(self.ifst)
        old_start = ofst.start
        new_start = ofst.add_state()
        ofst.add_arc(new_start, Arc(self.disambig, self.tag, 0.0, old_start))
        ofst.set_start(new_start)
        new_final = ofst.add_state()
        for s in ofst.finals:
            ofst.add_arc(s, Arc(self.disambig, self.tag, 0.0, new_final))
            ofst.states[s].weight = 0.0
        ofst.finals = []
        ofst.set_final(new_final)
        ofst.write_txt_fst(fst_file)


class EntitySlot(Slot):

    def __init__(self, index, name, config):
        super().__init__(index, name, config)
        self.read_list(config)

    def read_list(self, config):
        self.entities = []
        vocab = set()
        with open(config, 'r', encoding='utf8') as fin:
            for line in fin:
                arrs = line.strip().split()
                self.entities.append(arrs)
                vocab.update(arrs)
        self.vocab = sorted(vocab)

    def write_slot_fst(self, fst_file):
        """
        Please refer "IMPROVED RECOGNITION OF CONTACT NAMES IN VOICE COMMANDS"
        P(w|c) = exp(-alpha)/|C|^(1-beta)
        P(w|c) = 1 for all w if (alpha,beta)=(0,1)
        """
        alpha = 0
        beta = 0.75
        prob = alpha + (1 - beta) * math.log(len(self.entities))
        with open(fst_file, 'w', encoding='utf8') as fout:
            fout.write('0 1 {} {} {}\n'.format(self.disambig, self.tag, prob))
            fout.write('2 3 {} {}\n'.format(self.disambig, self.tag))
            fout.write('3\n')
            cur_state = 4
            for item in self.entities:
                for i, w in enumerate(item):
                    src = 1 if i == 0 else cur_state
                    dst = 2 if i == len(item) - 1 else cur_state + 1
                    fout.write('{} {} {} {}\n'.format(src, dst, w, w))
                    cur_state += 1


class Slots:

    def __init__(self, slot_config_file=None):
        self.read_slots(slot_config_file)

    def read_slots(self, slot_config_file=None):
        # Mapping entity to int id
        self.slots = []
        if slot_config_file is None:
            return
        # 0 is kept for <eps>
        # 1 is kept for $root
        # Please refer http://www.openfst.org/twiki/bin/view/FST/ReplaceDoc
        slot_id = 2
        vocab = set()
        with open(slot_config_file, 'r', encoding='utf8') as fin:
            for line in fin:
                if line.startswith('#'):
                    continue
                arr = line.strip().split()
                if len(arr) == 0:
                    continue
                assert len(arr) == 2
                name = arr[0]
                config_file = arr[1]
                slot = self.read_slot(slot_id, name, config_file)
                self.slots.append(slot)
                vocab.update(slot.vocab)
                slot_id += 1
        self.vocab = sorted(vocab)

    def read_slot(self, index, name, config_file):
        config_file = os.path.expandvars(config_file)
        if config_file == '{}':
            return EmptySlot(index, name, config_file)
        elif config_file.endswith('.fst.txt'):
            return FstSlot(index, name, config_file)
        else:
            return EntitySlot(index, name, config_file)

    def num_slots(self):
        return len(self.slots)

    def tags(self):
        return [x.tag for x in self.slots]

    def replaces(self):
        return [x.replace for x in self.slots]

    def disambigs(self):
        return [x.disambig for x in self.slots]

    def write_fst_and_replace(self, symbols, ifst, ofst):
        base_name = os.path.dirname(os.path.abspath(ofst))
        replace_cmd = 'fstreplace --epsilon_on_replace {} 1'.format(ifst)
        print('Creating slot fst ...')
        for slot in self.slots:
            if not slot.is_empty:
                txt_fst_file = os.path.join(base_name,
                                            '{}.fst.txt'.format(slot.name))
                fst_file = os.path.join(base_name, '{}.fst'.format(slot.name))
                slot.write_slot_fst(txt_fst_file)
                cmd = 'fstcompile --isymbols={} --osymbols={} {} > {}'.format(
                    symbols, symbols, txt_fst_file, fst_file)
                print(cmd)
                os.system(cmd)
                replace_cmd += ' {} {}'.format(fst_file, slot.index)
        replace_cmd += ' > {}'.format(ofst)
        print('Replace on G ...')
        print(replace_cmd)
        os.system(replace_cmd)
