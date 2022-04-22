"""
Experimenting with loading raw data
"""

# <codecell>
from music21.converter import parse
from data import *

# <codecell>
# score = parse('data/lmd_full/c/c7f1ab95981ff33f2e96590205e5b1bc.mid')

# %%
scores = load_composer('ryansMammoth')


# %%
score = scores[12]
score.write('midi', 'tmp.mid')

# %%
def load_all():
    composers = [
        ('bach', 0.5), 
        ('beethoven', 0.5), 
        ('mozart', 0.5), 
        ('ryansMammoth', 0.25)
    ]

    all_ds = {}
    for name, quant in composers:
        print('processing', name)
        scores = load_composer(name)
        ds = scores_to_dataset(scores, sampling_rate=quant)
        all_ds[name] = ds
    
    return all_ds

all_ds = load_all()

# %%

# %%
