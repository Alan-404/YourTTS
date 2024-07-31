import os
import numpy as np
import json
from pydub import AudioSegment
import librosa
from typing import Union, Optional, List, Tuple
import re
import pickle
import torch
import torch.nn.functional as F

MAX_AUDIO_VALUE = 32768

class VITSProcessor:
    def __init__(self, 
                 path: str, pad_token: str = "<PAD>", delim_token: str = "|", unk_token: str = "<UNK>", puncs: str = r"([:./,?!@#$%^&=`~;*\(\)\[\]\"\\])",
                 sampling_rate: int = 22050, num_mels: int = 80, n_fft: int = 1024, hop_length: int = 256, win_length: int = 1024, fmin: float = 0.0, fmax: float = 8000.0,
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
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_mels = num_mels
        self.win_length = win_length

        self.mel_basis = torch.from_numpy(
            librosa.filters.mel(
                sr=sampling_rate,
                n_fft=n_fft,
                n_mels=num_mels,
                fmin=fmin,
                fmax=fmax
            )
        ).to(device)

        self.hann_window = torch.hann_window(window_length=win_length).to(device)

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

        return patterns

    def read_pickle(self, path: str) -> np.ndarray:
        with open(path, 'rb') as file:
            signal = pickle.load(file)
            
        signal = librosa.resample(y=signal, orig_sr=8000, target_sr=self.sampling_rate)

        return signal
    
    def read_pcm(self, path: str) -> np.ndarray:
        audio = AudioSegment.from_file(path, frame_rate=8000, channels=1, sample_width=2).set_frame_rate(self.sampling_rate).get_array_of_samples()
        return np.array(audio).astype(np.float64) / MAX_AUDIO_VALUE
    
    def read_signal(self, path: str, role: Optional[int] = None) -> np.ndarray:
        if role is not None:
            signal, _ = librosa.load(path, sr=self.sampling_rate, mono=False)
            signal = signal[role]
        else:
            signal, _ = librosa.load(path, sr=self.sampling_rate, mono=True)

        return signal
    
    def spectral_normalize(self, x: torch.Tensor, C: int = 1, clip_val: float = 1e-5) -> torch.Tensor:
        return torch.log(torch.clamp(x, min=clip_val) * C)

    def mel_spectrogram(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.device != self.device:
            signal = signal.to(self.device)
        print(signal.shape)
        signal = F.pad(signal.unsqueeze(1), (int((self.n_fft-self.hop_length)/2), int((self.n_fft-self.hop_length)/2)), mode='reflect')
        print(signal.shape)
        signal = signal.squeeze(1)
        print(signal.shape)
        spec = torch.stft(signal, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.hann_window,
                      center=False, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
        print(spec.shape)
        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
        print(spec.shape)
        print(self.mel_basis.shape)
        spec = torch.matmul(self.mel_basis, spec)
        spec = self.spectral_normalize(spec)

        return spec
    
    def split_segment(self, signal: torch.Tensor, start: float, end: float):
        return signal[int(start * self.sampling_rate) : int(end * self.sampling_rate)]

    def load_audio(self, path: str, start: Optional[float] = None, end: Optional[float] = None, role: Optional[int] = None) -> torch.Tensor:
        if ".pickle" in path:
            signal = self.read_pickle(path)
        elif ".pcm" in path:
            signal = self.read_pcm(path)
        else:
            signal = self.read_signal(path, role)

        if start is not None and end is not None:
            signal = self.split_segment(signal, start, end)

        signal = torch.tensor(signal).to(self.device)
        # if signal.device != self.device:
        #     signal = signal.to(self.device)
        return signal
    
    def split_signal(self, signal: np.ndarray, threshold_length_segment_max: float = 60.0, threshold_length_segment_min: float = 0.1):
        intervals = []

        for top_db in range(30, 5, -5):
            intervals = librosa.effects.split(signal, top_db=top_db, frame_length=4096, hop_length=1024)
            if len(intervals) != 0 and max((intervals[:, 1] - intervals[:, 0]) / self.sampling_rate) <= threshold_length_segment_max:
                break
            
        return np.array([i for i in intervals if threshold_length_segment_min < (i[1] - i[0]) / self.sampling_rate <= threshold_length_segment_max])

    def clean_text(self, sentence: str) -> str:
        sentence = str(sentence)
        sentence = re.sub(self.puncs, "", sentence)
        sentence = re.sub(r"\s\s+", " ", sentence)
        sentence = sentence.strip()
        return sentence
    
    def slide_graphemes(self, text: str, patterns: List[str], n_grams: int = 3, reverse: bool = True):
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
    
    def sentence2phonemes(self, sentence: str):
        sentence = self.clean_text(sentence.upper())
        words = sentence.split(" ")
        graphemes = []

        length = len(words)

        for index, word in enumerate(words):
            graphemes += self.slide_graphemes(self.spec_replace(word), self.pattern, n_grams=4)
            if index != length - 1:
                graphemes.append(self.delim_token)

        return graphemes
    
    def find_token_id(self, token: str):
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
    
    def sentence2tokens(self, sentence: str):
        phonemes = self.sentence2phonemes(sentence)
        tokens = self.phonemes2tokens(phonemes)
        return tokens

    def phonemes2tokens(self, phonemes: List[str]):
        tokens = []
        for phoneme in phonemes:
            tokens.append(self.find_token_id(phoneme))
        return torch.tensor(tokens)
    
    def __call__(self, items: List[List[str]]) -> Tuple[torch.Tensor]:
        tokens = []
        lengths = []
        max_length = 0
        for phonemes in items:
            length = len(phonemes)
            lengths.append(length)
            if max_length < length:
                max_length = length

            tokens.append(self.phonemes2tokens(phonemes))

        padded_tokens = []
        for index, token_list in enumerate(tokens):
            padded_tokens.append(F.pad(token_list, (0, max_length - lengths[index]), mode='constant', value=self.pad_id))
        return torch.stack(padded_tokens), torch.tensor(lengths)
    
    def as_target(self, signals: List[torch.Tensor]):
        lengths = []
        max_length = 0

        for signal in signals:
            length = len(signal)
            lengths.append(length)
            if max_length < length:
                max_length = length
        
        padded_signals = []
        for index, signal in enumerate(signals):
            padded_signals.append(
                F.pad(signal, (0, max_length - lengths[index]), value=0.0)
            )

        return torch.stack(padded_signals), torch.tensor(lengths)
    