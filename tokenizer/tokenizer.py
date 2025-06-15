from datasets import load_dataset
import json
import os
import torch
from typing import List, Tuple
from tqdm import tqdm
import sys
import os

# Add the parent directory (attention) to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))  # tokenizer/
parent_dir = os.path.dirname(current_dir)                 # attention/
sys.path.append(parent_dir)

# Now you can import directly
from data.wmt14_loader import WMTDataset


class BPETokenizer():
    def __init__(self, dataset=None, max_seq_len = 128,  min_freq = 2, vocab_size = 37000):
        self.dataset = dataset
        self.vocab_size = vocab_size
        self.vocab = {} 
        self.min_freq = min_freq
        self.max_seq_len = max_seq_len
        self.bos_token_id = None
        self.pad_token_id = None
        self.eos_token_id = None
        self.unk_token_id = None
        self.special_tokens = None
        self.freq_pair = {}

    def __call__(self, sentences, padding=True, max_length=512, return_tensors=True, truncation=True):
        tokens = self.tokenize(sentences)
        ids = self.tokens_to_ids(tokens)
        shifted_ids = ids.copy()
        for idx, _ in enumerate(tokens):
            ids[idx] = [self.bos_token_id] + ids[idx] + [self.eos_token_id]
            if len(ids[idx]) > max_length:
                ids[idx] = ids[idx][:max_length]
                ids[idx][-1] = self.eos_token_id
            shifted_ids[idx] = ids[idx][1:]
            ids[idx] = ids[idx][:-1]
            if padding:
                ids[idx] = ids[idx] + [self.pad_token_id] * (max_length - len(ids[idx]))
                shifted_ids[idx] = shifted_ids[idx] + [self.pad_token_id] * (max_length - len(shifted_ids[idx]))
        if return_tensors:
            return torch.tensor(ids), torch.tensor(shifted_ids), (torch.tensor(ids) != self.pad_token_id).long()
        return ids, shifted_ids, (ids != self.pad_token_id)



    def decode(self, token_ids, skip_special_tokens=True):
        if type(token_ids) == list : token_ids = torch.tensor(token_ids)
        if token_ids.dim() == 1: token_ids = token_ids.unsqueeze(0)
        id_to_token = {v: k for k, v in self.vocab.items()}
        decoded_tokens = []
        for token_id in token_ids:
            if skip_special_tokens:
                tokens = [id_to_token[i.item()]for i in token_id if i not in self.special_tokens]
            else:
                tokens = [id_to_token[i.item()] for i in token_id]
            decoded_tokens.append(''.join(tokens).replace('Ġ', ' ').strip())
        return decoded_tokens
    
    
    def update_freq_pair(self, data_tokens):
        for tokens in data_tokens:
            for idx in range(1, len(tokens)):
                if tokens[idx-1] == '[UNK]' or tokens[idx] == '[UNK]':
                    continue
                pair = (tokens[idx-1], tokens[idx])
                if pair not in self.freq_pair:
                    self.freq_pair[pair] = 0
                self.freq_pair[pair] += 1

    def build(self):
        special_tokens = [
            "[PAD]",  
            "<s>",    
            "</s>",   
            "[UNK]",  
            "[MASK]",  
            "<en>",    
            "<de>",
            "Ġ"  
        ]
        BASIC_PUNCTUATION = [
            '.', ',', '!', '?', ';', ':', 
            '(', ')', '[', ']', '{', '}',
            '"', "'", '`'
        ]
        basic_tokens = [chr(i) for i in range(ord('a'), ord('z') + 1)] + ['ä', 'ö', 'ü', 'ß']
        for token in special_tokens + basic_tokens + BASIC_PUNCTUATION:
            self.vocab[token] = len(self.vocab)
        self.bos_token_id = self.vocab["<s>"]
        self.pad_token_id = self.vocab["[PAD]"]
        self.unk_token_id = self.vocab["[UNK]"]
        self.eos_token_id = self.vocab["</s>"]
        self.special_tokens = [
            self.bos_token_id, self.pad_token_id, self.unk_token_id, self.eos_token_id
        ]
        max_iterations = 1000  # Set a maximum number of iterations to prevent infinite loops
        de_data_tokens = []
        en_data_tokens = []
        for sample in self.dataset:
            de, en = sample[0], sample[1]
            de_tokens = self.tokenize(de)
            en_tokens = self.tokenize(en)
            de_data_tokens.append(de_tokens)
            en_data_tokens.append(en_tokens)
        for iteration_count in tqdm(range(max_iterations)):
            self.freq_pair = {}
            self.update_freq_pair(de_data_tokens)
            self.update_freq_pair(en_data_tokens) 
            self.update_vocab()
            de_data_tokens = self.re_merge(de_data_tokens)
            en_data_tokens = self.re_merge(en_data_tokens)
            if iteration_count % 100 == 0:
                print(f'At {iteration_count+1} Vocab {len(self.vocab)}')
            if len(self.vocab) >= self.vocab_size:
                return

    def re_merge(self, data_tokens):
        for i, _ in enumerate(data_tokens):
            idx = 0
            while idx+1 < len(data_tokens[i]):
                if data_tokens[i][idx] + data_tokens[i][idx+1] in self.vocab:
                    data_tokens[i][idx] = data_tokens[i][idx] + data_tokens[i][idx+1]
                    del data_tokens[i][idx+1]
                    idx = max(0, idx-1)
                else:
                    idx += 1
        return data_tokens

    def update_vocab(self):
        for idx, pair in enumerate(sorted(self.freq_pair.items(), key=lambda item : item[1], reverse=True)):
            if idx == 100 or len(self.vocab) == self.vocab_size or pair[1] < self.min_freq:
                return
            new_token = pair[0][0] + pair[0][1]
            self.vocab[new_token] = len(self.vocab)



    def tokenize(self, sentences: List[str]):
        if type(sentences) == str:
            sentences = [sentences]
        sentences = [sentence.lower().replace(' ', 'Ġ') for sentence in sentences]
        tokens = []
        for j, sentence in enumerate(sentences):
            tokens.append([])
            for char in sentence: 
                try:
                    if char in self.vocab:
                        tokens[j].append(char)
                except:
                    print(char)
                    print('Decoding Error')
                    input()
            idx = 0
            while idx < len(tokens[j]) - 1:
                merged_token = tokens[j][idx] + tokens[j][idx + 1]
                if merged_token in self.vocab:
                    tokens[j][idx] = merged_token
                    del tokens[j][idx + 1]
                    idx = max(idx-1, 0)
                else:
                    idx += 1
        return tokens if len(tokens) > 1 else tokens[0]


    def tokens_to_ids(self, sentences):
        if type(sentences[0]) != list:
            sentences = [sentences]
        sentences_ids = []
        for sentence in sentences:
            ids = []
            for token in sentence:
                if token in self.vocab:
                    ids.append(self.vocab[token])
                else:
                    ids.append(self.vocab["[UNK]"])
            sentences_ids.append(ids) 
        return sentences_ids if len(sentences_ids) > 1 else sentences_ids[0]
    
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        # Save vocab
        with open(os.path.join(path, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        # Save config (other attributes)
        config = {
            "max_seq_len": self.max_seq_len,
            "min_freq": self.min_freq,
            "vocab_size": self.vocab_size,
            "bos_token_id": self.bos_token_id,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
            "unk_token_id": self.unk_token_id,
            "special_tokens" : self.special_tokens
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path, dataset=None):
        # Load vocab
        with open(os.path.join(path, "vocab.json"), "r", encoding="utf-8") as f:
            vocab = json.load(f)
        # Load config
        with open(os.path.join(path, "config.json"), "r") as f:
            config = json.load(f)
        # Create new instance
        tokenizer = cls(dataset=dataset,
                        max_seq_len=config["max_seq_len"],
                        min_freq=config["min_freq"],
                        vocab_size=config["vocab_size"],
                        )
        tokenizer.bos_token_id = config['bos_token_id']
        tokenizer.pad_token_id = config['pad_token_id']
        tokenizer.eos_token_id = config['eos_token_id']
        tokenizer.unk_token_id = config['unk_token_id']
        tokenizer.special_tokens = config['special_tokens']
        tokenizer.vocab = vocab
        return tokenizer


if __name__ == '__main__':
    dataset = WMTDataset(
        split='train[:100_000]', 
        min_length=256,      
        max_length=512,     
        max_length_ratio=2.5, 
        remove_duplicates=True,
        normalize_unicode=True
    )
    tokenizer = BPETokenizer(dataset, min_freq=2, vocab_size=10_000)
    tokenizer.build()
    tokenizer.save('WMTTokenizer_v1')
    print(f"Vocabulary size: {len(tokenizer.vocab)}")


