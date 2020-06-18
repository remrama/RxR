"""
1. Run a logistic regression, predicting task completion.
2. Summary stats across participants who did/didn't visit.
3. Plot baseline lucid dreaming frequency and visit success.
"""
from os import path
from json import load

import pandas as pd
import pingouin as pg

import matplotlib.pyplot as plt; plt.ion()
from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = 'Arial'
rcParams['axes.labelsize'] = 'x-large'
rcParams['xtick.labelsize'] = 'medium'
rcParams['ytick.labelsize'] = 'medium'
rcParams['xtick.major.size'] = 5
rcParams['ytick.major.size'] = 5
rcParams['ytick.minor.size'] = 3
rcParams['legend.fontsize'] = 'medium'
rcParams['legend.title_fontsize'] = 'medium'
rcParams['savefig.dpi'] = 300
rcParams['svg.fonttype'] = 'none'


with open('./config.json','r') as jsonfile:
    p = load(jsonfile)
    data_dir  = path.expanduser(p['data_directory'])
    deriv_dir = path.expanduser(p['derivatives_directory'])
    LuCiD_factors = p['LuCiD_factors']
    colors = p['colors']

LDQ_PREDICTORS = ['ld_freq']

export_stats_fname   = path.join(deriv_dir,'visit_predictors.tsv')
export_summary_fname = path.join(deriv_dir,'visit_summary.tsv')
export_plot_fname    = path.join(deriv_dir,'visit_plot.svg')

participants_fname = path.join(data_dir,'participants.tsv')
LuCiD_fname        = path.join(data_dir,'phenotype','LuCiD.tsv')
LDQ_fname          = path.join(data_dir,'phenotype','LDQ.tsv')



# load participant file to find out who visited the room
participants_df = pd.read_csv(participants_fname,sep='\t',index_col='participant_id')

# load the baseline LDQ file to find out previous LD frequencies
LDQ_df = pd.read_csv(LDQ_fname,sep='\t',index_col='participant_id')

# load the baseline LuCiD file to find out trait lucidity
LuCiD_df = pd.read_csv(LuCiD_fname,sep='\t',index_col='participant_id')

# get the simplified LuCiD factors
for factor, values in LuCiD_factors.items():
    factor_cols = [ f'LuCiD_{v}' for v in values ]
    LuCiD_df[factor] = LuCiD_df[factor_cols].mean(axis=1)

# combine all dataframes to make things easier.
df = pd.concat([participants_df,LDQ_df,LuCiD_df],axis='columns')

# just keep the columns of interest
keep_cols = ['reinstatement'] + LDQ_PREDICTORS + list(LuCiD_factors.keys())
df = df[keep_cols]


#### test if either prev LD frequency
#### or LuCiD factors predict task completion (room visit)

# binarize room visit success
df['visit_bool'] = df['reinstatement'].apply(lambda x: x!='False')
# df['visit'] = df['visit'].apply(lambda x: x in ['lucid','semi-lucid'])

predictors = LDQ_PREDICTORS + ['control']
# predictors = ['ld_freq'] + list(LuCiD_factors.keys())
stats = pg.logistic_regression(df[predictors],df['visit_bool'])
stats.to_csv(export_stats_fname,float_format='%.03f',index=False,sep='\t')

# export summary stats dataframe
summary_df = df.groupby('reinstatement').describe()
summary_df.columns = [ '-'.join(c) for c in summary_df.columns ]
summary_df.to_csv(export_summary_fname,float_format='%.01f',index=True,sep='\t')



####### plot



categories = ['False','non-lucid','semi-lucid','lucid']
df['ld_freq'] = pd.Categorical(df['ld_freq'],categories=range(8),ordered=True)
df['reinstatement'] = pd.Categorical(df['reinstatement'],categories=categories,ordered=True)
freqs = df['ld_freq'].value_counts()

plot_data = pd.crosstab(df['ld_freq'],df['reinstatement'],dropna=False)
plot_data['semi-lucid'] = plot_data[['non-lucid','semi-lucid']].sum(axis=1)
plot_data['lucid'] = plot_data[['semi-lucid','lucid']].sum(axis=1)

LINEWIDTH = 1

_, ax = plt.subplots(figsize=(6,6))

x = freqs.index.values
y = freqs.values
ax.bar(x,y,color='white',label='None',
    edgecolor='k',linewidth=LINEWIDTH)

x = plot_data.index.values
for z, label in enumerate(categories[1:]):
    c = colors[label]
    y = plot_data[label].values
    ax.bar(x,y,color=c,zorder=10-z,label=label,
           edgecolor='k',linewidth=LINEWIDTH)

ax.legend(title='Dream reinstatement',frameon=False,loc='upper left')


ax.set_ylabel('Number of participants')
ax.set_xlabel('How often do you have lucid dreams?')
ax.set_xticks(range(8))
ax.set_xticklabels(['never','<1 per year','1 per year','2-4 per year','1 per month','2-3 times per month','1 per week','>1 per week'],
                   rotation=30,ha='right',va='top')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(export_plot_fname)
