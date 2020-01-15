"""
Make txt file that aggregates relevant dream reports.
"""
from os import path
from json import load

from pandas import read_csv


DATA_DIR  = path.join('..','data')
DERIV_DIR = path.join(DATA_DIR,'derivatives')

out_txt_fname = path.join(DERIV_DIR,'dream_reports.txt')

participants_fname = path.join(DATA_DIR,'participants.tsv')
participants_df = read_csv(participants_fname,sep='\t',index_col='participant_id')

dream_reports = ''
# loop through partipants and grab dream report if they made the visit
for sub, visit in participants_df['reinstatement'].items():

    if visit != 'False':
        basename = f'{sub}_ses-home_report-red.json'
        report_fname = path.join(DATA_DIR,sub,basename)

        with open(report_fname,'r') as jsonfile:
            data = load(jsonfile)
            report_txt = data['Describe your dream']

        dream_reports += f'{visit} - {sub}\n\t\t{report_txt}\n'

with open(out_txt_fname,'w') as txtfile:
    txtfile.write(dream_reports.rstrip('\n'))