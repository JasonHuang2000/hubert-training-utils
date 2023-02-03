import glob
import logging
import os
import random
from typing import Dict

from tqdm import tqdm
import fire

from fairseq.data.audio.audio_utils import get_features_or_waveform

def main(
    data_root: str, # path to 'LibriSpeech' directory
    output_dir: str,
    split: str = '100', # should be one of '100', '360', '500', or '960'
    speaker_sample_ratio: float = 0.2,
    data_sample_ratio: float = 0.5,
):
    logging.basicConfig(level=logging.INFO)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir)
    data_root = os.path.abspath(data_root)
    split = str(split)
    
    split_mapping: Dict[str, str] = {
        '100': 'train-clean-100',
        '360': 'train-clean-360',
        '500': 'train-other-500',
    }

    if split == '960':
        speakers = []
        for _, dir in split_mapping.items():
            spk_list = os.listdir(os.path.join(data_root, dir))
            speakers += [os.path.join(dir, spk) for spk in spk_list]
    else:
        spk_list = os.listdir(os.path.join(data_root, split_mapping[split]))
        speakers = [os.path.join(split_mapping[split], spk) for spk in spk_list]

    sampled_speakers = random.sample(speakers, int(len(speakers)*speaker_sample_ratio))
    logging.info("Total number of speakers: %d. Sampling %d speakers. (sampling rate = %s)", len(speakers), len(sampled_speakers), speaker_sample_ratio)

    os.chdir(data_root)

    with open(os.path.join(output_dir, "train.tsv"), "w") as f:
        audio_paths = []
        for speaker in sampled_speakers:
            paths = glob.glob(f"{speaker}/**/*.flac", recursive=True)
            audio_paths += random.sample(paths, int(len(paths)*data_sample_ratio))
        logging.info("%d audio files have been sampled. (sampling rate = %s)", len(audio_paths), data_sample_ratio)

        logging.info("Writing results to %s...", output_path)
        f.write(f"{data_root}\n")
        lines = []
        for audio_path in tqdm(audio_paths):
            wav = get_features_or_waveform(audio_path, need_waveform=True, use_sample_rate=16000)
            lines.append(f"{audio_path}\t{len(wav)}\n")
        f.writelines(lines)

    with open(os.path.join(output_dir, "valid.tsv"), "w") as f:
        for audio_path in tqdm(glob.glob("dev-clean/**/*.flac", recursive=True)):
            wav = get_features_or_waveform(audio_path, need_waveform=True, use_sample_rate=16000)
            lines.append(f"{audio_path}\t{len(wav)}\n")
        f.writelines(lines)

if __name__ == "__main__":
    fire.Fire(main)

