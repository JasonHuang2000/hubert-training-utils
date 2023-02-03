import os

import fire
import librosa
from tqdm import tqdm

def main(
    meta_file: str,
):
    with open(meta_file, 'r') as f:
        lines = f.readlines()
    
    data_root = lines[0][:-1]
    audio_paths = [line.split('\t')[0] for line in lines[1:]]

    total_seconds: float = .0
    for audio_path in tqdm(audio_paths):
        total_seconds += librosa.get_duration(filename=os.path.join(data_root, audio_path))

    print(f"Audio duration for {meta_file}: {total_seconds:.3f} seconds | {total_seconds/60:.3f} minutes | {total_seconds/60/60:.3f} hours")

if __name__ == "__main__":
    fire.Fire(main)
