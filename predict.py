# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import random

# We need to set `TRANSFORMERS_CACHE` before any imports, which is why this is up here.
MODEL_PATH = "/src/models/"
os.environ["TRANSFORMERS_CACHE"] = MODEL_PATH
os.environ["TORCH_HOME"] = MODEL_PATH

from typing import Optional
from cog import BasePredictor, Input, Path

# Model specific imports
import torchaudio
import typing as tp
import numpy as np

import torch

from audiocraft.solvers.compression import CompressionSolver

from audiocraft.models import MusicGen, MultiBandDiffusion
from audiocraft.solvers.compression import CompressionSolver
from audiocraft.models.loaders import (
    load_compression_model,
    load_lm_model,
)
from audiocraft.data.audio import audio_write

from audiocraft.models.builders import get_lm_model
from omegaconf import OmegaConf

from BeatNet.BeatNet import BeatNet
import librosa

import subprocess

def _delete_param(cfg, full_name: str):
    parts = full_name.split('.')
    for part in parts[:-1]:
        if part in cfg:
            cfg = cfg[part]
        else:
            return
    OmegaConf.set_struct(cfg, False)
    if parts[-1] in cfg:
        del cfg[parts[-1]]
    OmegaConf.set_struct(cfg, True)

def load_ckpt(path, device):
    loaded = torch.load(str(path))
    cfg = OmegaConf.create(loaded['xp.cfg'])
    cfg.device = str(device)
    if cfg.device == 'cpu':
        cfg.dtype = 'float32'
    else:
        cfg.dtype = 'float16'
    _delete_param(cfg, 'conditioners.self_wav.chroma_chord.cache_path')
    _delete_param(cfg, 'conditioners.self_wav.chroma_stem.cache_path')
    _delete_param(cfg, 'conditioners.args.merge_text_conditions_p')
    _delete_param(cfg, 'conditioners.args.drop_desc_p')

    lm = get_lm_model(loaded['xp.cfg'])
    lm.load_state_dict(loaded['model']) 
    lm.eval()
    lm.cfg = cfg
    compression_model = CompressionSolver.wrapped_model_from_checkpoint(cfg, cfg.compression_model_checkpoint, device=device)
    return MusicGen(f"{os.getenv('COG_USERNAME')}/musicgen-chord", compression_model, lm)

class Predictor(BasePredictor):
    def setup(self, weights: Optional[Path] = None):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = load_ckpt('musicgen_chord.th', self.device)
        self.model.lm.condition_provider.conditioners['self_wav'].match_len_on_eval = True
        self.mbd = MultiBandDiffusion.get_mbd_musicgen()

        self.beatnet = BeatNet(
            1,
            mode="offline",
            inference_model="DBN",
            plot=[],
            thread=False,
            device="cuda:0",
        )


    def _load_model(
        self,
        model_path: str,
        cls: Optional[any] = None,
        load_args: Optional[dict] = {},
        model_id: Optional[str] = None,
        device: Optional[str] = None,
    ) -> MusicGen:

        if device is None:
            device = self.device

        compression_model = load_compression_model(
            model_id, device=device, cache_dir=model_path
        )
        lm = load_lm_model(model_id, device=device, cache_dir=model_path)
        
        return MusicGen(model_id, compression_model, lm)

    def predict(
        self,
        prompt: str = Input(
            description="A description of the music you want to generate.", default=None
        ),
        music_input: Path = Input(
            description="An audio file input for the remix.",
            default=None,
        ),
        multi_band_diffusion: bool = Input(
            description="If `True`, the EnCodec tokens will be decoded with MultiBand Diffusion.",
            default=False,
        ),
        normalization_strategy: str = Input(
            description="Strategy for normalizing audio.",
            default="loudness",
            choices=["loudness", "clip", "peak", "rms"],
        ),
        top_k: int = Input(
            description="Reduces sampling to the k most likely tokens.", default=250
        ),
        top_p: float = Input(
            description="Reduces sampling to tokens with cumulative probability of p. When set to  `0` (default), top_k sampling is used.",
            default=0.0,
        ),
        temperature: float = Input(
            description="Controls the 'conservativeness' of the sampling process. Higher temperature means more diversity.",
            default=1.0,
        ),
        classifier_free_guidance: int = Input(
            description="Increases the influence of inputs on the output. Higher values produce lower-varience outputs that adhere more closely to inputs.",
            default=3,
        ),
        output_format: str = Input(
            description="Output format for generated audio.",
            default="wav",
            choices=["wav", "mp3"],
        ),
        seed: int = Input(
            description="Seed for random number generator. If `None` or `-1`, a random seed will be used.",
            default=None,
        ),
    ) -> Path:

        if prompt is None:
            raise ValueError("Must provide `prompt`.")
        if not music_input:
            raise ValueError("Must provide `music_input`.")
        
        if prompt is None:
             prompt = ''
         
        model = self.model

        set_generation_params = lambda duration: model.set_generation_params(
            duration=duration,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            cfg_coef=classifier_free_guidance,
        )

        if not seed or seed == -1:
            seed = torch.seed() % 2 ** 32 - 1
            set_all_seeds(seed)
        set_all_seeds(seed)
        print(f"Using seed {seed}")

        y, sr = librosa.load(music_input)

        from bpm_detector import bpm_detector
        tempo, _ = bpm_detector(y, sr)
        tempo = tempo[0]
        # tempo = librosa.beat.beat_track(y=music_input, sr=sr) #not that accurate
        print(f"BPM : {int(tempo)}")
        
        prompt = prompt + f', bpm : {int(tempo)}'
        
        bpm_detector = None

        music_input, sr = torchaudio.load(music_input)
        music_input = music_input[None] if music_input.dim() == 2 else music_input
        duration = music_input.shape[-1]/sr

        vocals = self.separate_vocals(music_input, sr)

        encodec_rate = 50
        sub_duration=15
        overlap = 30 - sub_duration
        wavs = []
        wav_sr = model.sample_rate
        set_generation_params(30)

        wav, tokens = model.generate_with_chroma(['the intro of ' + prompt], music_input[...,:30*sr], sr, progress=True, return_tokens=True)
        if multi_band_diffusion:
            wav = self.mbd.tokens_to_wav(tokens)
        wavs.append(wav.detach().cpu())
        audio_write(
            "wav_0",
            wav[0].cpu(),
            model.sample_rate,
            strategy=normalization_strategy,
        )
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        for i in range(int((duration - overlap) // sub_duration) - 1):
            wav, tokens = model.generate_continuation_with_audio_tokens(
            prompt=tokens[...,sub_duration*encodec_rate:],
            melody_wavs = music_input[...,sub_duration*(i+1)*sr:(sub_duration*(i+1)+30)*sr],
            melody_sample_rate=sr,
            descriptions=['chorus of ' + prompt],
            progress=True,
            return_tokens=True,
            )
            if multi_band_diffusion:
                wav = self.mbd.tokens_to_wav(tokens)
            wavs.append(wav.detach().cpu())
            audio_write(
                f"wav_{i+1}",
                wav[0].cpu(),
                model.sample_rate,
                strategy=normalization_strategy,
            )
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        if int(duration - overlap) % sub_duration != 0:
            set_generation_params(overlap + ((duration - overlap) % sub_duration)) ## 여기
            wav, tokens = model.generate_continuation_with_audio_tokens(
                prompt=tokens[...,sub_duration*encodec_rate:],
                melody_wavs = music_input[...,sub_duration*(len(wavs))*sr:],
                melody_sample_rate=sr,
                descriptions=['the outro of ' + prompt],
                progress=True,
                return_tokens=True,
            )
            if multi_band_diffusion:
                wav = self.mbd.tokens_to_wav(tokens)
            wavs.append(wav.detach().cpu())
            audio_write(
                "wav_last",
                wav[0].cpu(),
                model.sample_rate,
                strategy=normalization_strategy,
            )
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        wav = wavs[0][...,:sub_duration*wav_sr].cpu()
        for i in range(len(wavs)-1):
            if i == len(wavs)-2:
                wav = torch.concat([wav,wavs[i+1]],dim=-1).cpu()
            else:
                wav = torch.concat([wav,wavs[i+1][...,:sub_duration*wav_sr]],dim=-1).cpu()
        # print(wav.shape, vocals.shape)
        wav = wav / np.abs(wav).max()
        audio_write(
            "background",
            wav[0].cpu(),
            model.sample_rate,
            strategy=normalization_strategy,
            loudness_compressor=True,
        )
        # beats = self.estimate_beats(wav, model.sample_rate)
        wav = wav.cpu() + vocals[...,:wav.shape[-1]].cpu()*0.3
        
        audio_write(
            "out",
            wav[0].cpu(),
            model.sample_rate,
            strategy=normalization_strategy,
            loudness_compressor=True,
        )
        wav_path = "out.wav"

        if output_format == "mp3":
            mp3_path = "out.mp3"
            subprocess.call(["ffmpeg", "-i", wav_path, mp3_path])
            os.remove(wav_path)
            path = mp3_path
        else:
            path = wav_path

        return Path(path)

    def _preprocess_audio(
        audio_path, model: MusicGen, duration: tp.Optional[int] = None
    ):

        wav, sr = torchaudio.load(audio_path)
        wav = torchaudio.functional.resample(wav, sr, model.sample_rate)
        wav = wav.mean(dim=0, keepdim=True)

        # Calculate duration in seconds if not provided
        if duration is None:
            duration = wav.shape[1] / model.sample_rate

        # Check if duration is more than 30 seconds
        if duration > 30:
            raise ValueError("Duration cannot be more than 30 seconds")

        end_sample = int(model.sample_rate * duration)
        wav = wav[:, :end_sample]

        assert wav.shape[0] == 1
        assert wav.shape[1] == model.sample_rate * duration

        wav = wav.cuda()
        wav = wav.unsqueeze(1)

        with torch.no_grad():
            gen_audio = model.compression_model.encode(wav)

        codes, scale = gen_audio

        assert scale is None

        return codes

    def estimate_beats(self, wav, sample_rate):
        # resample to BeatNet's sample rate
        beatnet_input = librosa.resample(
            wav,
            orig_sr=sample_rate,
            target_sr=self.beatnet.sample_rate,
        )
        return self.beatnet.process(beatnet_input)

    def separate_vocals(self, music_input, sr):
        from demucs.audio import convert_audio
        from demucs.apply import apply_model

        wav = convert_audio(music_input, sr, self.model.lm.condition_provider.conditioners['self_wav'].demucs.samplerate, self.model.lm.condition_provider.conditioners['self_wav'].demucs.audio_channels)
        stems = apply_model(self.model.lm.condition_provider.conditioners['self_wav'].demucs, wav, device=self.device)
        vocals = stems[:, self.model.lm.condition_provider.conditioners['self_wav'].demucs.sources.index('vocals')]
        vocals = convert_audio(vocals, self.model.lm.condition_provider.conditioners['self_wav'].demucs.samplerate, self.model.sample_rate, 1)
        return vocals
    
# From https://gist.github.com/gatheluck/c57e2a40e3122028ceaecc3cb0d152ac
def set_all_seeds(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
