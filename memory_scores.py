"""
Get waking memory scores
"""
from os import path
from json import load

import pandas as pd
import numpy as np

with open('./config.json','r') as jsonfile:
    p = load(jsonfile)
    data_dir  = path.expanduser(p['data_directory'])
    deriv_dir = path.expanduser(p['derivatives_directory'])

participants_fname = path.join(data_dir,'participants.tsv')

participants_df = pd.read_csv(participants_fname,sep='\t',index_col='participant_id')


def find_memory_score(row):
    sub = row.name
    memory_fname = path.join(data_dir,sub,f'{sub}_ses-lab_eat-red.tsv')
    # sub 20 doesn't have the EAT form
    if not path.exists(memory_fname):
        return np.nan
    else:
        memory_df = pd.read_csv(memory_fname,sep='\t')
        memory_df['item'] = np.repeat(range(8),2)

        # sub 18 didn't fill out back of form so lots of NaNs
        memory_df.dropna(subset=['accuracy'],inplace=True)
        memory_df['accuracy'] = memory_df['accuracy'].astype(bool)

        # groupy by each item and say they remember the item
        # if they get at least one of the 2 probes right for that item
        memory_score = memory_df.groupby('item')['accuracy'].mean(
            ).astype(bool).mean()

        return memory_score


participants_df['memory'] = participants_df.apply(find_memory_score,axis=1)

# export a small dataframe that shows the each participant individually
participants_df[ ['reinstatement','memory']
    ].to_csv(path.join(deriv_dir,'memory_scores-by_subj.tsv'),index=True,sep='\t',na_rep='NaN')

# average/describe for each reinstatement group
describe_df = participants_df.groupby('reinstatement')['memory'].describe()
# add a row that has the whole group
all_series = participants_df['memory'].describe()
all_series.name = 'all'
describe_df = describe_df.append(all_series)

describe_df.round(2
    ).to_csv(path.join(deriv_dir,'memory_scores-descr.tsv'),index=True,sep='\t',float_format='%.02f',na_rep='NaN')
