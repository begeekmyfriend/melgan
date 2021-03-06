import os
import glob
import tqdm
import torch
import argparse
import numpy as np
from apex import amp
from scipy.io.wavfile import write
from model.generator import Generator
from utils.hparams import HParam, load_hparam_str


MAX_WAV_VALUE = 32768.0

def main(args):
    checkpoint = torch.load(args.checkpoint_path)
    if args.config is not None:
        hp = HParam(args.config)
    else:
        hp = load_hparam_str(checkpoint['hp_str'])

    model = Generator(hp.audio.n_mel_channels).cuda()
    model.load_state_dict(checkpoint['model_g'])
    model.eval()
    if hp.train.amp:
        model, _ = amp.initialize(model, [], opt_level=hp.train.amp_level)

    with torch.no_grad():
        for melpath in tqdm.tqdm(glob.glob(os.path.join(args.input_folder, '*.npy'))):
            mel = torch.from_numpy(np.load(melpath))
            if len(mel.shape) == 2:
                mel = mel.unsqueeze(0)
            mel = mel.cuda()

            audio = model.inference(mel)
            audio = audio.cpu().numpy()

            fname = os.path.splitext(os.path.basename(melpath))[0]
            out_path = 'melgan_' + fname + '_epoch%04d.wav' % checkpoint['epoch']
            write(out_path, hp.audio.sampling_rate, audio)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/default.yaml',
                        help="yaml file for config. will use hp_str from checkpoint if not given.")
    parser.add_argument('-p', '--checkpoint_path', type=str, required=True,
                        help="path of checkpoint pt file for evaluation")
    parser.add_argument('-i', '--input_folder', type=str, default='',
                        help="directory of mel-spectrograms to invert into raw audio. ")
    args = parser.parse_args()

    main(args)
