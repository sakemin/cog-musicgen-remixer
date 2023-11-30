import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from .btc.mir_eval import *
from .btc.utils import chords
from .btc.btc_model import *
from .btc.utils.mir_eval_modules import audio_file_to_features, idx2chord, idx2voca_chord, get_audio_paths

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import torchaudio
import librosa

import argparse
import warnings

class ChordExtractor(nn.Module):

    def __init__(self, device, sample_rate, max_duration, chroma_len, n_chroma, winhop):
        super().__init__()
        self.config = HParams.load("/src/audiocraft/modules/btc/run_config.yaml") #gotta specify the path for run_config.yaml of btc

        # self.config.feature['large_voca'] = False
        # self.config.model['num_chords'] = 25

        self.model_file = '/src/audiocraft/modules/btc/test/btc_model_large_voca.pt'
        # self.model_file = 'audiocraft/modules/btc/test/btc_model.pt'
        self.idx_to_chord = idx2voca_chord()
        self.sr = sample_rate

        self.n_chroma = n_chroma
        self.max_duration = max_duration
        self.chroma_len = chroma_len
        self.to_timebin = self.max_duration/self.chroma_len
        self.timebin = winhop

        self.chords = chords.Chords()
        self.device = device

        self.denoise_window_size = 7
        self.denoise_threshold = 0.5
        
        self.model = BTC_model(config=self.config.model).to(device)
        if os.path.isfile(self.model_file):
            checkpoint = torch.load(self.model_file)
            self.mean = checkpoint['mean']
            self.std = checkpoint['std']
            self.model.load_state_dict(checkpoint['model'])

    def forward(self, wavs:torch.Tensor) -> torch.Tensor:
        sr = self.config.mp3['song_hz']
        chromas = []
        for wav in wavs:
            original_wav = librosa.resample(wav.cpu().numpy(), orig_sr=self.sr, target_sr=sr)
            original_wav = original_wav.squeeze(0)
            # print(original_wav.shape)
            T = original_wav.shape[-1]
            # in case we are getting a wav that was dropped out (nullified)
            # from the conditioner, make sure wav length is no less that nfft
            if T <  self.timebin//4:
                pad = self.timebin//4 - T
                r = 0 if pad % 2 == 0 else 1
                original_wav = F.pad(torch.Tensor(original_wav), (pad // 2, pad // 2 + r), 'constant', 0)
                original_wav = original_wav.numpy()
                assert original_wav.shape[-1] == self.timebin//4, f"expected len {self.timebin//4} but got {original_wav.shape[-1]}"
            # print(original_wav.shape)
            #preprocess
            currunt_sec_hz = 0

            while len(original_wav) > currunt_sec_hz + self.config.mp3['song_hz'] * self.config.mp3['inst_len']:
                start_idx = int(currunt_sec_hz)
                end_idx = int(currunt_sec_hz + self.config.mp3['song_hz'] * self.config.mp3['inst_len'])
                tmp = librosa.cqt(original_wav[start_idx:end_idx], sr=sr, n_bins=self.config.feature['n_bins'], bins_per_octave=self.config.feature['bins_per_octave'], hop_length=self.config.feature['hop_length'])
                if start_idx == 0:
                    feature = tmp
                else:
                    feature = np.concatenate((feature, tmp), axis=1)
                currunt_sec_hz = end_idx
            
            if currunt_sec_hz == 0:
                feature = librosa.cqt(original_wav[currunt_sec_hz:], sr=sr, n_bins=self.config.feature['n_bins'], bins_per_octave=self.config.feature['bins_per_octave'], hop_length=self.config.feature['hop_length'])
            else:
                tmp = librosa.cqt(original_wav[currunt_sec_hz:], sr=sr, n_bins=self.config.feature['n_bins'], bins_per_octave=self.config.feature['bins_per_octave'], hop_length=self.config.feature['hop_length'])
                feature = np.concatenate((feature, tmp), axis=1)
            # print(feature.shape)
            feature = np.log(np.abs(feature) + 1e-6)
            # print(feature)
            feature_per_second = self.config.mp3['inst_len'] / self.config.model['timestep']
            song_length_second = len(original_wav)/self.config.mp3['song_hz']

            feature = feature.T
            feature = (feature - self.mean)/self.std

            time_unit = feature_per_second
            n_timestep = self.config.model['timestep']

            num_pad = n_timestep - (feature.shape[0] % n_timestep)
            feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
            num_instance = feature.shape[0] // n_timestep

            #inference
            start_time = 0.0
            lines = []
            with torch.no_grad():
                self.model.eval()
                feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(self.device)
                for t in range(num_instance):
                    self_attn_output, _ = self.model.self_attn_layers(feature[:, n_timestep * t:n_timestep * (t + 1), :])
                    prediction, _ = self.model.output_layer(self_attn_output)
                    prediction = prediction.squeeze()
                    for i in range(n_timestep):
                        if t == 0 and i == 0:
                            prev_chord = prediction[i].item()
                            continue
                        if prediction[i].item() != prev_chord:
                            lines.append(
                                '%.3f %.3f %s\n' % (start_time, time_unit * (n_timestep * t + i), self.idx_to_chord[prev_chord]))
                            start_time = time_unit * (n_timestep * t + i)
                            prev_chord = prediction[i].item()
                        if t == num_instance - 1 and i + num_pad == n_timestep:
                            if start_time != time_unit * (n_timestep * t + i):
                                lines.append('%.3f %.3f %s\n' % (start_time, time_unit * (n_timestep * t + i), self.idx_to_chord[prev_chord]))
                            break

            strlines = ''.join(lines)

            chroma = []

            count = 0
            for line in lines:
                if count >= self.chroma_len: 
                    break
                splits = line.split()
                if len(splits) == 3:
                    s = splits[0]
                    e = splits[1]
                    l = splits[2]

                crd = self.chords.chord(l)
                
                if crd[0] == -1:
                    multihot = torch.Tensor(crd[2])
                else:
                    multihot = torch.concat([torch.Tensor(crd[2])[-crd[0]:],torch.Tensor(crd[2])[:-crd[0]]])
                start_bin = round(float(s)/self.to_timebin)
                end_bin = round(float(e)/self.to_timebin)
                for j in range(start_bin,end_bin):
                    if count >= self.chroma_len: 
                        break
                    chroma.append(multihot)
                    count += 1
            
            chroma = torch.stack(chroma, dim=0)

            # Denoising chroma
            kernel = torch.ones(self.denoise_window_size)/self.denoise_window_size

            filtered_signals = []
            for i in range(chroma.shape[-1]):
                filtered_signals.append(torch.nn.functional.conv1d(chroma[...,i].unsqueeze(0),
                                                        kernel.unsqueeze(0).unsqueeze(0).to(chroma.device), 
                                                        padding=(self.denoise_window_size - 1) // 2))
            filtered_signals = torch.stack(filtered_signals, dim=-1)
            filtered_signals = filtered_signals > self.denoise_threshold

            chromas.append(filtered_signals.squeeze(0))
        
        return torch.stack(chromas, dim=0).to(self.device)