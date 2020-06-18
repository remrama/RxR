"""
Get the length of time between lab visit
and dream visit for successful cases.
"""
from os import path
from json import load

import pandas as pd

with open('./config.json','r') as jsonfile:
    p = load(jsonfile)
    data_dir  = path.expanduser(p['data_directory'])
    deriv_dir = path.expanduser(p['derivatives_directory'])

participants_fname = path.join(data_dir,'participants.tsv')

participants_df = pd.read_csv(participants_fname,sep='\t',index_col='participant_id')

participants_df['acq_time'] = pd.to_datetime(participants_df['acq_time'])

def find_date_difference(row):
    if row['reinstatement'] == 'False':
        return pd.NA
    else:
        # load the reports json and get the dream visit date
        sub = row.name
        report_fname = path.join(data_dir,sub,f'{sub}_ses-home_report-red.json')
        with open(report_fname,'r') as jsonfile:
            data = load(jsonfile)
            dreamvisit_time = pd.to_datetime(data['Date and time'])

        labvisit_time = row['acq_time']
        time_difference = dreamvisit_time - labvisit_time

        return time_difference.days


participants_df['days_between'] = participants_df.apply(find_date_difference,axis=1)

participants_df.dropna(subset=['days_between'],inplace=True)
participants_df['days_between'] = participants_df['days_between'].astype(int)

# export a small dataframe that shows the days between for participants who visited
participants_df[ ['reinstatement','days_between']
    ].to_csv(path.join(deriv_dir,'days_between_visits-by_subj.tsv'),index=True,sep='\t')

# save out descriptives
participants_df.groupby('reinstatement')['days_between'].describe().round(2
    ).to_csv(path.join(deriv_dir,'days_between_visits-descr.tsv'),index=True,sep='\t',float_format='%.02f')
