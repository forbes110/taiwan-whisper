import argparse
import opencc
import csv
import editdistance
from tqdm import tqdm
from collections import defaultdict
from functools import partial
from pypinyin import pinyin, lazy_pinyin, Style
from g2p_en import G2p # too slow, should use lexicon instead
lexicon_fpath = './lexicon.lst'


class MixErrorRate(object):
    def __init__(self, to_simplified_chinese=True, to_traditional_chinese=False, phonemize=False):
        self.converter = None
        if to_simplified_chinese and to_traditional_chinese:
            raise ValueError("Can't convert to both simplified and traditional chinese at the same time.")
        if to_simplified_chinese:
            print("Convert to simplified chinese")
            self.converter = opencc.OpenCC('t2s.json')
        elif to_traditional_chinese:
            print("Convert to traditional chinese")
            self.converter = opencc.OpenCC('s2t.json')
        else:
            print("No chinese conversion")
        if phonemize:
            print("Phonemize chinese and english words")
            print("Force traditional to simplified conversion")
            self.converter = opencc.OpenCC('t2s.json')
            self.zh_phonemizer = partial(lazy_pinyin, style=Style.BOPOMOFO, errors='ignore')
            self.zh_bopomofo_stress_marks = ['ˊ', 'ˇ', 'ˋ', '˙']
            # self.en_phonemizer = G2p()
            # self.en_valid_phonemes = [p for p in self.en_phonemizer.phonemes]
            # for p in self.en_phonemizer.phonemes:
            #     if p[-1].isnumeric():
            #         self.en_valid_phonemes.append(p[:-1])
            # use lexicon instead
            self.en_wrd2phn = defaultdict(lambda: [])
            with open(lexicon_fpath, 'r', encoding='utf-8') as f:
                for line in f:
                    word, phonemes = line.strip().split('\t')
                    self.en_wrd2phn[word] = phonemes.split()
        self.phonemize = phonemize
    
    def _from_str_to_list(self, cs_string):
        cs_list = []
        cur_en_word = ''
        for s in cs_string:
            # if is space, skip it
            if s in [' ', '\t', '\n', '\r', ',', '.', '!', '?', '。', '，', '！', '？', '、', '；', '：', '「', '」', '『', '』', '（', '）', '(', ')', '\[', '\]', '{', '}', '<', '>', '《', '》', '“', '”', '‘', '’', '…', '—', '～', '·', '•']:
                if cur_en_word != '':
                    cs_list.append(cur_en_word)
                    cur_en_word = ''
                continue
            # if it chinese character, add it to list
            if u'\u4e00' <= s <= u'\u9fff':
                if cur_en_word != '':
                    cs_list.append(cur_en_word)
                    cur_en_word = ''
                if self.converter is not None:
                    s = self.converter.convert(s)
                cs_list.append(s)
            # check character, if it is english character, add it to current word
            elif s.isalnum() or s in ["'", "-"]:
                cur_en_word += s
            else:
                print(f"Unknown character during conversion: {s}")
        if cur_en_word != '':
            cs_list.append(cur_en_word)
        return cs_list

    def _phonemized_cs_list(self, cs_list):
        cur_zh_chars = []
        phonemes = []
        for unit in cs_list:
            if u'\u4e00' <= unit[0] <= u'\u9fff':
                cur_zh_chars.append(unit)
            else:
                if cur_zh_chars:
                    zh_phns = ''.join(self.zh_phonemizer(''.join(cur_zh_chars)))
                    phonemes.extend(filter(lambda p: p not in self.zh_bopomofo_stress_marks, zh_phns))
                    cur_zh_chars = []
                phonemes.extend(self.en_wrd2phn[unit])
        if cur_zh_chars:
            zh_phns = ''.join(self.zh_phonemizer(''.join(cur_zh_chars)))
            phonemes.extend(filter(lambda p: p not in self.zh_bopomofo_stress_marks, zh_phns))
            cur_zh_chars = []
        return phonemes

    def compute(self, predictions=None, references=None, show_progress=True, empty_error_rate=1.0, **kwargs):
        total_err = 0
        total_ref_len = 0
        iterator = tqdm(enumerate(zip(predictions, references)), total=len(predictions), desc="Computing Mix Error Rate...") if show_progress and len(predictions) > 100 else enumerate(zip(predictions, references))
        for i, (pred, ref) in iterator:
            # if english use word error rate, if chinese use character error rate
            # generate list for editdistance computation
            pred_list = self._from_str_to_list(pred)
            ref_list = self._from_str_to_list(ref)
            if self.phonemize:
                pred_list = self._phonemized_cs_list(pred_list)
                ref_list = self._phonemized_cs_list(ref_list)
            # compute edit distance
            err = editdistance.eval(pred_list, ref_list)
            total_err += err
            total_ref_len += len(ref_list)
        if total_ref_len == 0:
            print(f"No reference found, return {empty_error_rate*100}% error rate instead")
            return empty_error_rate # if no reference, return 100% error rate instead
        return total_err / total_ref_len # mer

def load_output_csv(fpath):
    with open(fpath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        predictions = []
        references = []
        columns = next(reader)
        print(f"Columns: {columns}")
        for i, row in enumerate(reader):
            # if i == 0:
                # continue
            predictions.append(row[4])
            references.append(row[3])
        return predictions, references

def main(args):
    print(args)
    mer = MixErrorRate(to_simplified_chinese=args.to_simplified_chinese, to_traditional_chinese=args.to_traditional_chinese)
    predictions, references = load_output_csv(args.csv_fpath)
    mer_value = mer.compute(predictions=predictions, references=references)
    print(f"Mix Error Rate: {mer_value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Mix Error Rate")
    parser.add_argument("--csv_fpath", type=str, required=True, help="Path to the csv file")
    parser.add_argument("--to_simplified_chinese", action="store_true", help="Convert chinese to simplified chinese")
    parser.add_argument("--to_traditional_chinese", action="store_true", help="Convert chinese to traditional chinese")
    args = parser.parse_args()
    main(args)
