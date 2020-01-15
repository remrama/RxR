"""
Plot LuCiD scales -- both state and trait next to each other.

- single plot per participant (that completed the task)
- one group plot of just the trait LuCiD
"""
from os import path
from json import load

import pandas as pd

from scipy.stats import wilcoxon

import seaborn as sea
import matplotlib.pyplot as plt; plt.ion()
from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = 'Arial'
rcParams['axes.labelsize'] = 'x-large'
rcParams['xtick.labelsize'] = 'x-large'
rcParams['ytick.labelsize'] = 'x-large'
rcParams['xtick.major.size'] = 5
rcParams['ytick.major.size'] = 5
rcParams['ytick.minor.size'] = 3
# rcParams['legend.fontsize'] = 'small'
# rcParams['legend.title_fontsize'] = 'medium'
rcParams['savefig.dpi'] = 300
rcParams['svg.fonttype'] = 'none'


with open('./config.json','r') as jsonfile:
    p = load(jsonfile)
    data_dir  = path.expanduser(p['data_directory'])
    deriv_dir = path.expanduser(p['derivatives_directory'])
    LuCiD_factors = p['LuCiD_factors']
    colors = p['colors']

participants_fname = path.join(data_dir,'participants.tsv')
participants_df = pd.read_csv(participants_fname,sep='\t',index_col='participant_id')

# for any subject that did complete the task,
# get trait LuCiD (all one file)
# and state LuCiD (individual files)
LuCiD_fname = path.join(data_dir,'phenotype','LuCiD.tsv')
trait_LuCiD = pd.read_csv(LuCiD_fname,sep='\t',index_col='participant_id')

# get the state LuCiDs (ie, corresponding to individual dream reports)
state_LuCiDs = {}
for sub, visit in participants_df['reinstatement'].items():
    if visit != 'False':
        basename = f'{sub}_ses-home_report-red.json'
        report_fname = path.join(data_dir,sub,basename)

        with open(report_fname,'r') as jsonfile:
            data = load(jsonfile)
            LuCiD_scores = data['LuCiD']

        state_LuCiDs[sub] = LuCiD_scores

state_LuCiD = pd.DataFrame.from_dict(state_LuCiDs,
        orient='index',
        dtype=float,
        columns=[ f'LuCiD_{i+1}' for i in range(28) ]
    ).replace({'N/A':pd.np.nan,'NaN':pd.np.nan})
state_LuCiD.index.name = 'participant_id'


# get the LuCiD factors for both state and trait

# just use subjects who complete task
participants = state_LuCiD.index.values
index = pd.MultiIndex.from_product(
    [participants,('state','trait')],names=['participant_id','survey'])
columns = LuCiD_factors.keys()
df = pd.DataFrame(columns=columns,index=index)
for factor, values in LuCiD_factors.items():
    factor_cols = [ f'LuCiD_{v}' for v in values ]
    df.loc[(participants,'state'),factor] = state_LuCiD.loc[participants,factor_cols].mean(axis=1).values
    df.loc[(participants,'trait'),factor] = trait_LuCiD.loc[participants,factor_cols].mean(axis=1).values
    
    ##### TEMP??
    # also add to trait df for group plot
    trait_LuCiD[factor] = trait_LuCiD[factor_cols].mean(axis=1)
# #     # state_LuCiD[f'state_{factor}'] = state_LuCiD[factor_cols].mean(axis=1)
# df.reset_index(drop=False,inplace=True)
# # # combine both dataframes
# # df = pd.concat([trait_LuCiD,state_LuCiD],axis='columns',sort=True
# #     ).reset_index(drop=False
# #     ).rename(columns={'index':'participant_id'})
# # value_cols = [ col for col in df.columns if 
# #     'state' in col or 'trait' in col ]
# # id_cols = ['participant_id']
# df = pd.melt(df,id_vars=['participant_id','survey'],
#     value_vars=LuCiD_factors.keys(),
#     var_name='factor',value_name='likert')
# #     ).dropna()
# # df['survey'] = df['factor'].apply(lambda x: x.split('_')[0])
# # df['factor'] = df['factor'].apply(lambda x: x.split('_')[1])

# ###### plot

# sea.catplot(data=df,x='factor',hue='survey',y='likert',col='participant_id',
#     kind='bar',col_wrap=3,hue_order=['trait','state'],
#     palette={'trait':'gainsboro','state':'royalblue'})


######## plot group trait

# trait_summary = df.loc[(slice(None),'trait'),columns].astype(float
#     ).groupby('survey'
#     ).agg(['mean','sem'])

trait_summary = trait_LuCiD[columns].agg(['mean','sem'])

# get columns sorted high to low for plot
ascending_cols = trait_summary.sort_values(
        by='mean',ascending=False,axis=1
    ).columns


# wilcoxon tests for LuCiD scores against middle
stats = { col: wilcoxon(trait_LuCiD[col].values - 2.5)
    for col in ascending_cols }
stats = pd.DataFrame.from_dict(stats,columns=['wilcoxon_stat','pval_2tailed'],orient='index')
export_stats_fname = path.join(deriv_dir,f'LuCiD-trait.tsv')
stats.to_csv(export_stats_fname,index=True,index_label='factor',float_format='%.05f',sep='\t')


_, ax = plt.subplots(figsize=(11,6))

x = range(len(ascending_cols))
y = trait_summary.loc['mean',ascending_cols].values
yerr = trait_summary.loc['sem',ascending_cols].values

ax.bar(x,y,color=colors['trait'],width=.8,zorder=0,alpha=1)
ax.errorbar(x,y,yerr,fmt='none',linewidth=1,color='k',capthick=1)

# boxes or violins
boxdata = trait_LuCiD[ascending_cols].values
WIDTHS = .6
VIOLIN_COLOR = colors['trait']
# violins = ax.violinplot(boxdata,positions=x,widths=WIDTHS,showextrema=False)
# plt.setp(violins['bodies'],facecolor=VIOLIN_COLOR,
#     edgecolor='k',alpha=1)

# xpos = x
boxpos = pd.np.array(x) - .25
BOXWIDTH = .2
LINEWIDTH = .5
ax.boxplot(boxdata,positions=boxpos,widths=BOXWIDTH,
    notch=True,showcaps=False,
    medianprops=dict(linewidth=LINEWIDTH,color='k'),
    boxprops=dict(linewidth=LINEWIDTH),
    whiskerprops=dict(linewidth=LINEWIDTH),
    flierprops=dict(marker='.',
                    markerfacecolor='k',
                    markeredgecolor='none'))

# # overlap individuals (for now)
# for sub, row in trait_LuCiD.iterrows():
#     y = row[ascending_cols].values
#     visit = participants_df.loc[sub,'visit']
#     c = colors[visit] if visit!='False' else 'gray'
#     ax.plot(x,y,linestyle='-',linewidth=.8,color=c,marker='.')

# significance markers
alpha_cutoff = .05
bonferonni_cutoff = alpha_cutoff / len(ascending_cols)

ymarker = 5.5
for i, factor in enumerate(ascending_cols):
    pval = stats.loc[factor,'pval_2tailed']
    if pval < alpha_cutoff:
        fill = 'full' if pval < bonferonni_cutoff else 'none'
        sigmarker = '*'
        ax.plot(i,ymarker,marker=sigmarker,
            markersize=9,markeredgewidth=.5,
            fillstyle=fill,color='k',clip_on=False)

ax.set_xticks(range(len(ascending_cols)))
xticklabels = [ x.replace('_',' ') for x in ascending_cols ]
ax.set_xticklabels(xticklabels,rotation=30,ha='right',va='top')
ax.set_xlabel('LuCiD factor (trait lucidity)')
ax.set_xlim(min(x)-.5,max(x)+.5)
ax.set_yticks([0,5])
ax.set_yticks([1,2,3,4],minor=True)
yticklabels = ['Strongly\ndisagree','Strongly\nagree']
ax.set_yticklabels(yticklabels)
ax.set_ylim(0,5)
ax.axhline(2.5,color='k',linestyle='--',linewidth=.5,zorder=-1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()

export_plot_fname = path.join(deriv_dir,f'LuCiD-trait.svg')
plt.savefig(export_plot_fname)
plt.close()






######## plot individuals

WIDTH = .3
dodge = WIDTH/2.

for sub, subdf in df.groupby('participant_id'):

    _, ax = plt.subplots(figsize=(7,3))

    xbase = pd.np.arange(len(columns))
    visit = participants_df.loc[sub,'reinstatement']

    for survey, row in subdf.groupby('survey'):
        y = row[list(columns)].values.flatten()
        x = xbase + dodge*(1 if survey == 'state' else -1)
        if survey == 'state':
            c = colors[visit]
        else:
            c = colors[survey]
        ax.bar(x,y,color=c,width=WIDTH)

    ax.set_xticks(range(len(columns)))
    xticklabels = [ x.replace('_',' ') for x in columns ]
    ax.set_xticklabels(xticklabels,rotation=30,ha='right',va='top')

    ax.set_yticks([0,5])
    ax.set_yticks([1,2,3,4],minor=True)
    yticklabels = ['Strongly\ndisagree','Strongly\nagree']
    ax.set_yticklabels(yticklabels)
    ax.set_ylim(0,5)

    # ax.axhline(2.5,color='k',linestyle='--',linewidth=.5,zorder=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    export_plot_fname = path.join(deriv_dir,'LuCiDs',f'LuCiD-{sub}.svg')
    plt.savefig(export_plot_fname)
    plt.close()