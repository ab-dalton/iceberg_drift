# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 15:01:02 2023

@author: adalt043
"""
    
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# -----------------------------------------------------------------------------
# Load database
# -----------------------------------------------------------------------------
    
# Load most recent Iceberg Beacon Database output file
df = pd.read_csv('D:/Abby/paper_2/Iceberg Beacon Database-20211026T184427Z-001/Iceberg Beacon Database/iceberg_beacon_database_17032023_no_talbot.csv', index_col=False)

# Convert to datetime
df["datetime_data"] = pd.to_datetime(df["datetime_data"].astype(str), format="%Y-%m-%d %H:%M:%S")

path_figures = "D:/Abby/paper_2/plots/"

df['beacon_id'].nunique()


#Create year, month, day columns
df['day'] = df['datetime_data'].dt.day
df['month'] = df['datetime_data'].dt.month
df['year'] = df['datetime_data'].dt.year

# Function to associate month numbers with season names
seasons = {12: 'Winter',
           1: 'Winter',
           2:'Winter',
           3: 'Spring',
           4: 'Spring',
           5:'Spring',
           6:'Summer',
           7:'Summer',
           8:'Summer',
           9:'Fall',
           10:'Fall',
           11:'Fall'}

df['Season'] = df['month'].apply(lambda x: seasons[x])

winter = df.loc[df['Season'] == "Winter"]
spring = df.loc[df['Season'] == "Spring"]
summer = df.loc[df['Season'] == "Summer"]
fall = df.loc[df['Season'] == "Fall"]





# params = {'mathtext.default': 'regular' }
# plt.rcParams.update(params)
# font = {'size'   : 12,
#         'weight' : 'normal'}
# mpl.rc('font', **font)
fig, axs = plt.subplots(2,2, figsize=(6, 4), constrained_layout=True, sharey=True, sharex=True)

sns.histplot(x="speed_ms",
             data=winter,
             binwidth=0.2,
             color="grey",
             edgecolor="black",
             ax=axs[0,0])
axs[0,0].set(xlabel=None, ylabel=None)
axs[0,0].annotate('A', (1, 1),
                    xytext=(-5,-5),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    ha='right', va='top',
                    fontsize=14)

sns.histplot(x="speed_ms",
             data=spring,
             binwidth=0.2,
             color="grey",
             edgecolor="black",
             ax=axs[0,1])
axs[0,1].set(xlabel=None, ylabel=None)
axs[0,1].annotate('B', (1, 1),
                    xytext=(-5,-5),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    ha='right', va='top',
                    fontsize=14)

sns.histplot(x="speed_ms",
             data=summer,
             binwidth=0.2,
             color="grey",
             edgecolor="black",
             ax=axs[1,0])
axs[1,0].set(xlabel=None, ylabel=None)
axs[1,0].annotate('C', (1, 1),
                    xytext=(-5,-5),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    ha='right', va='top',
                    fontsize=14)

sns.histplot(x="speed_ms",
             data=fall,
             binwidth=0.2,
             color="grey",
             edgecolor="black",
             ax=axs[1,1])
axs[1,1].set(xlabel=None, ylabel=None)
axs[1,1].annotate('D', (1, 1),
                    xytext=(-5,-5),
                    xycoords='axes fraction',
                    textcoords='offset points',
                    ha='right', va='top',
                    fontsize=14)

fig.text(0.5, -0.03, 'Speed (m $s^{-1}$)', ha='center', va='center', fontsize=12)
fig.text(-0.03, 0.5, 'Count', ha='center', va='center', rotation='vertical', fontsize=12)


# Save figure
fig.savefig(
    path_figures + "seasonal_speed.png",
    dpi=300,
    transparent=False,
    bbox_inches="tight",
)








# # Create a 2x2 grid of subplots
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8), constrained_layout=True)

# # Set plot parameters
# params = {'mathtext.default': 'regular' }   
# plt.rcParams.update(params)
# font = {'size': 12, 'weight': 'normal'}
# mpl.rc('font', **font)

# # Loop over the seasons and plot each histogram in a separate subplot
# for i, season in enumerate(['Winter', 'Spring', 'Summer', 'Fall']):
#     row = i // 2
#     col = i % 2
#     ax = axes[row, col]
#     ax.set(xlabel="Speed (m $s^{-1}$)", ylabel="Count")
#     sns.histplot(x="speed_ms", data=df[df['Season'] == season], binwidth=0.1, color="grey", edgecolor="black", ax=ax)
#     # ax.set_title(season, loc='center')
#     ax.set(ylim=(0,18000))
#     ax.set(xlim=(0,None))
#     ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))

# # Add space between subplots
# plt.tight_layout()
# # Show the plot
# plt.show()

# # Save figure
# fig.savefig(
#     path_figures + "seasonal_speed.png",
#     dpi=300,
#     transparent=False,
#     bbox_inches="tight",
# )









# for season in sorted(set(seasons.values())):
#     fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
#     params = {'mathtext.default': 'regular' }   
#     plt.rcParams.update(params)
#     font = {'size'   : 12,
#             'weight' : 'normal'}
#     mpl.rc('font', **font)
#     ax.set(xlabel="Speed (m $s^{-1}$)", ylabel="Count")
#     sns.histplot(
#         x="speed_ms", data=df[df['Season']==season], binwidth=0.1, color="grey", edgecolor="black", ax=ax
#     )
#     plt.title(season)
#     plt.show()
    



# fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
# params = {'mathtext.default': 'regular' }
# plt.rcParams.update(params)
# font = {'size'   : 12,
#         'weight' : 'normal'}
# mpl.rc('font', **font)
# ax.set(xlabel="Speed (m $s^{-1}$)", ylabel="Count")
# sns.histplot(x="speed_ms", data=df, binwidth=0.1, color="grey", edgecolor="black", ax=ax)
# plt.show()













