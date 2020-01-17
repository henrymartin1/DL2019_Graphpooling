# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 16:37:54 2020

@author: henry
"""

import pickle
import os
import os.path as osp
import sys
import warnings
from datetime import datetime
import json
import time
import pickle

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from experiments.complexity.complexity_config import config

results_dict = pickle.load( open( config['output_file'], "rb" ) )

figure_path = os.path.join('experiments', 'complexity')

time_tracking_df = results_dict['time']
memory_tracking_df = results_dict['memory']

# plot
sns.set(style="whitegrid")

# time complexity bar
g = sns.catplot(x="n_epochs", y="time", hue="method", data=time_tracking_df,
                height=6, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("Time in seconds")

g.savefig(os.path.join(figure_path,'complexity_time_bar.pdf'))
plt.close(g.fig)

# accuracy over time
g = sns.catplot(x="n_epochs", y="accuracy", hue="method", data=time_tracking_df,
                height=6, kind="point", palette="muted")
g.despine(left=True)
g.set_ylabels("Time in seconds")

g.savefig(os.path.join(figure_path,'complexity_accuracy_line.pdf'))
plt.close(g.fig)


# memory bar
g = sns.catplot(x="method", y="memory", data=memory_tracking_df,
                height=6, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("Time in seconds")

g.savefig(os.path.join(figure_path,'complexity_memory.pdf'))
plt.close(g.fig)


