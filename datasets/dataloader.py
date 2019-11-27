import os
import glob
import torch
import random
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader


def create_dataloader(hp, args, train):
    dataset = MelFromDisk(hp, args, train)

    if train:
        return DataLoader(dataset=dataset, batch_size=hp.train.batch_size, shuffle=True,
            num_workers=hp.train.num_workers, pin_memory=True, drop_last=True)
    else:
        return DataLoader(dataset=dataset, batch_size=1, shuffle=False,
            num_workers=hp.train.num_workers, pin_memory=True, drop_last=False)


class MelFromDisk(Dataset):
    def __init__(self, hp, args, train):
        self.hp = hp
        self.args = args
        self.train = train
        self.path = hp.data.train if train else hp.data.validation
        self.wav_list = glob.glob(os.path.join(self.path, '**', '*.wav'), recursive=True)
        self.mel_segment_length = hp.audio.segment_length // hp.audio.hop_length
        self.mel_pad_short = hp.audio.pad_short // hp.audio.hop_length
        self.mapping = [i for i in range(len(self.wav_list))]

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        if self.train:
            idx1 = idx
            idx2 = self.mapping[idx1]
            return self.my_getitem(idx1), self.my_getitem(idx2)
        else:
            return self.my_getitem(idx)

    def shuffle_mapping(self):
        random.shuffle(self.mapping)

    def my_getitem(self, idx):
        wavpath = self.wav_list[idx]
        melpath = wavpath.replace('.wav', '.npy')
        audio, sr = librosa.core.load(wavpath, self.hp.audio.sampling_rate)
        audio = torch.from_numpy(audio).unsqueeze(0)
        mel = torch.from_numpy(np.load(melpath))

        if audio.size(1) < self.hp.audio.segment_length + self.hp.audio.pad_short:
            mel = np.pad(mel, ((0, 0), (0, self.mel_segment_length + self.mel_pad_short - mel.size(1))), mode='constant', constant_values=-self.hp.audio.mel_bias)
            audio = np.pad(audio, ((0, 0), (0, self.hp.audio.segment_length + self.hp.audio.pad_short - audio.size(1))), mode='constant', constant_values=0.0)

        assert(self.hp.audio.hop_length * mel.size(1) == audio.size(1))

        if self.train:
            max_mel_start = mel.size(1) - self.mel_segment_length
            assert(max_mel_start > 0)
            mel_start = random.randint(0, max_mel_start)
            mel_end = mel_start + self.mel_segment_length
            mel = mel[:, mel_start:mel_end]

            audio_start = mel_start * self.hp.audio.hop_length
            audio = audio[:, audio_start:audio_start+self.hp.audio.segment_length]

        audio = audio + (1/32768) * torch.randn_like(audio)
        return mel, audio
