"""
Data loading utilities

author: William Tong (wtong@g.harvard.edu)
"""

from pathlib import Path
import pickle

import numpy as np
from music21 import corpus
from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset

REST_IDX = 128

def load_composer(data_dir='data/bach', name='bach'):
    if type(data_dir) == str:
        data_dir = Path(data_dir)
    
    scores = None
    if not data_dir.exists():
        data_dir.mkdir()
        bundle = corpus.search(name, 'composer')
        scores = [metadata.parse() for metadata in tqdm(bundle)]
        with (data_dir / f'{name}.pkl').open('wb') as fp:
            pickle.dump(scores, fp)
    else:
        with (data_dir / f'{name}.pkl').open('rb') as fp:
            scores = pickle.load(fp)
    
    return scores


def scores_to_dataset(scores):
    examples = [_batch(part) for score in scores for part in score.parts]
    examples = np.concatenate(examples, axis=0)
    dataset = TensorDataset(torch.tensor(examples))
    return dataset


def _batch(score, sampling_rate=0.25, bars=2):
    text = _to_text(score, sampling_rate=sampling_rate)
    text = np.concatenate(text, axis=0)

    # samps_per_batch = int(4 / sampling_rate * bars)
    samps_per_batch = 32
    total_samps = text.shape[0] // samps_per_batch

    batch = np.zeros((total_samps, samps_per_batch, 129))
    for i in range(total_samps):
        start_idx = i * samps_per_batch
        end_idx = (i + 1) * samps_per_batch
        # print('BATCH', batch[i,:,:].shape)
        # print('TEXT', text[start_idx:end_idx,:].shape)
        batch[i,:,:] = text[start_idx:end_idx,:]

    return batch.astype('float32')


def _to_text(score, sampling_rate) -> list:
    notes = score.flat.getElementsByClass(nt.Note)
    hist = _bin(notes, sampling_rate)
    end = score.flat.highestOffset

    text = [_to_word(hist[i]) for i in np.arange(0, end, sampling_rate)]
    return text

def _bin(notes, sampling_rate) -> defaultdict:
    hist = defaultdict(list)

    for note in notes:
        offset = note.offset
        halt = offset + note.duration.quarterLength

        if _precise_round(offset % sampling_rate) != 0:
            offset = _precise_round(offset - (offset % sampling_rate))
        if _precise_round(halt % sampling_rate) != 0:
            halt = _precise_round(halt + (sampling_rate - halt % sampling_rate))

        while offset < halt:
            hist[offset].append(note)
            offset += sampling_rate

    return hist

def _to_word(notes) -> str:
    if len(notes) == 0:
        one_hot = np.zeros((1, 129))
        one_hot[0, REST_IDX] = 1
        return one_hot

    # arp notes that occur together
    # TODO: might be interesting to arp across parts
    ordered_notes = sorted(notes, key=lambda n: n.pitch.midi, reverse=False)
    one_hot = np.zeros((len(ordered_notes), 129))
    for i, note in enumerate(ordered_notes):
        one_hot[i, note.pitch.midi] = 1

    return one_hot

def _precise_round(val, precision=10):
    return round(val * precision) / precision

