"""
Export dataframe of participant counts.
"""
from os import path
from json import load

import pandas as pd

with open('./config.json','r') as jsonfile:
    p = load(jsonfile)
    data_dir  = path.expanduser(p['data_directory'])
    deriv_dir = path.expanduser(p['derivatives_directory'])

in_fname = path.join(data_dir,'participants.tsv')
out_fname = path.join(deriv_dir,'participant_summary.tsv')

df = pd.read_csv(in_fname,sep='\t',index_col='participant_id')

df.groupby(['sex','compensation']).size(
    ).rename('frequency'
    ).to_csv(out_fname,header=True,sep='\t')