import os
import numpy as np
import json
import librosa
from typing import Union, Optional, List, Tuple
import re
import torch
import torch.nn.functional as F

from scipy.io import wavfile

MAX_AUDIO_VALUE = 32768

class YourTTSProcessor:
    def __init__(self, 
                 path: str, pad_token: str = "<PAD>", delim_token: str = "|", unk_token: str = "<UNK>", puncs: str = r"([:./,?!@#$%^&=`~;*\(\)\[\]\"\\])",
                 sampling_rate: int = 22050,
                 device: Union[str, int] = 'cpu') -> None:
        # Text Process
        assert os.path.exists(path)
        
        patterns = json.load(open(path, 'r', encoding='utf8'))
        self.yolo = patterns
        self.replace_dict = patterns['replace']
        self.mapping = patterns['mapping']
        self.single_vowels = patterns['single_vowel']
        vocab = []

        for key in patterns.keys():
            if key == 'replace' or key == 'mapping':
                continue
            vocab += patterns[key]
            
        self.dictionary = self.create_vocab_dictionary(vocab, pad_token, delim_token, unk_token)

        self.pattern = self.sort_pattern(vocab + list(patterns['mapping'].keys()))

        self.pad_token = pad_token
        self.delim_token = delim_token
        self.unk_token = unk_token

        self.pad_id = self.find_token_id(pad_token)
        self.delim_id = self.find_token_id(delim_token)
        self.unk_id = self.find_token_id(unk_token)
        
        # Audio Process
        self.sampling_rate = sampling_rate

        self.puncs = puncs
        self.device = device
    
    def create_vocab_dictionary(self, vocab: List[str], pad_token: str, delim_token: str, unk_token: str):
        dictionary = []
        dictionary.append(pad_token)

        for item in vocab:
            if item not in dictionary:
                dictionary.append(item)

        dictionary.append(delim_token)
        dictionary.append(unk_token)

        return dictionary

    def sort_pattern(self, patterns: List[str]):
        patterns = sorted(patterns, key=len)
        patterns.reverse()

    def clean_text(self, sentence: str) -> str:
        sentence = str(sentence)
        sentence = re.sub(self.puncs, "", sentence)
        sentence = re.sub(r"\s\s+", " ", sentence)
        sentence = sentence.strip()
        return sentence
    
    def slide_graphemes(self, text: str, patterns: List[str], n_grams: int = 3, reverse: bool = False):
        if len(text) == 1:
            if text in patterns:
                if text in self.mapping:
                    return [self.mapping[text]]
                else:
                    return [text]
            return [self.unk_token]
        if reverse:
            text = [text[i] for i in range(len(text) - 1, -1, -1)]
            text = "".join(text)
        graphemes = []
        start = 0
        if len(text) - 1 < n_grams:
            n_grams = len(text)
        num_steps = n_grams
        while start < len(text):
            found = True
            item = text[start:start + num_steps]

            if reverse:
                item = [item[i] for i in range(len(item) - 1, -1, -1)]
                item = "".join(item)
                
            if item in patterns:
                if item in self.mapping:
                    graphemes.append(self.mapping[item])
                else:
                    graphemes.append(item)
            elif num_steps == 1:
                graphemes.append(self.unk_token)
            else:
                found = False

            if found:
                start += num_steps
                if len(text[start:]) < n_grams:
                    num_steps = len(text[start:])
                else:
                    num_steps = n_grams
            else:
                num_steps -= 1

        if reverse:
            graphemes = [graphemes[i] for i in range(len(graphemes) - 1, -1, -1)]

        return graphemes
    
    def sentence2phonemes(self, sentence: str) -> List[str]:
        sentence = self.clean_text(sentence.upper())
        words = sentence.split(" ")
        graphemes = []

        length = len(words)

        for index, word in enumerate(words):
            graphemes += self.slide_graphemes(self.spec_replace(word), self.pattern, n_grams=4, reverse=False)
            if index != length - 1:
                graphemes.append(self.delim_token)

        return graphemes
    
    def find_token_id(self, token: str) -> int:
        if token in self.dictionary:
            return self.dictionary.index(token)
        return self.dictionary.index(self.unk_token)
    
    def token2text(self, tokens: np.ndarray, get_string: bool = False) -> str:
        words = []
        for token in tokens:
            words.append(self.dictionary[token])

        if get_string:
            return "".join(words).replace(self.delim_token, " ")
        
        return words
    
    def spec_replace(self, word: str) -> str:
        for key in self.replace_dict:
            arr = word.split(key)
            if len(arr) == 2:
                if arr[1] in self.single_vowels:
                    return word
                else:
                    return word.replace(key, self.replace_dict[key])
        return word
    
    def sentence2tokens(self, sentence: str) -> torch.Tensor:
        phonemes = self.sentence2phonemes(sentence)
        tokens = self.phonemes2tokens(phonemes)
        return tokens

    def phonemes2tokens(self, phonemes: List[str]):
        tokens = []
        for phoneme in phonemes:
            tokens.append(self.find_token_id(phoneme))
        return torch.tensor(tokens)
    
    # Condition Audio Hanlding
    def load_audio(self, path: str) -> torch.Tensor:
        sr, signal = wavfile.read(path)
        signal = signal / MAX_AUDIO_VALUE
        if sr != self.sampling_rate:
            signal = librosa.resample(signal, orig_sr=sr, target_sr=self.sampling_rate)
        return torch.tensor(signal, dtype=torch.float)
    
    def __call__(self, tokens: List[torch.Tensor], signals: List[torch.Tensor]):
        max_token_length = 0
        max_cond_length = 0

        token_lengths = []
        cond_lengths = []

        num_items = len(tokens)

        for i in range(num_items):
            # Token
            token_length = len(tokens[i])
            token_lengths.append(token_length)
            if token_length > max_token_length:
                max_token_length = token_length

            # Condition
            cond_length = len(signals[i])
            cond_lengths.append(cond_length)
            if cond_length > max_cond_length:
                max_cond_length = cond_length

        padded_tokens = []
        padded_conds = []

        for i in range(num_items):
            padded_tokens.append(
                F.pad(tokens[i], (0, max_token_length - token_lengths[i]), value=self.pad_id)
            )
            padded_conds.append(
                F.pad(signals[i], (0, max_cond_length - cond_lengths[i]), value=0.0)
            )

        padded_tokens = torch.stack(padded_tokens)
        padded_conds = torch.stack(padded_conds)

        token_lengths = torch.tensor(token_lengths)
        cond_lengths = torch.tensor(cond_lengths)

        return padded_tokens, padded_conds, token_lengths, cond_lengths