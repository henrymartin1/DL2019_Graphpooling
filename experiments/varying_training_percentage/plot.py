import os
import os.path as osp
import sys
import warnings
from datetime import datetime
import json
import csv
import time
import random
import re

import pandas as pd
import numpy as np
import seaborn as sns


#w = csv.writer(open("results.csv", "w"))
#for key, val in train_percentage_dict.items():
#    w.writerow([key, val])
#print("All done")

train_percentage_dict = {   'method': [],
                            'train_percentage': [],
                            'best_accuracy': []}

with open('results.csv', mode='r') as infile:
    reader = csv.reader(infile)
    line = 0
    for row in reader:
        if line==0:
            print(row[1])
            for char in row[1]:
                if char in " []\'":
                    row[1] = row[1].replace(char,'')
            print("\n\n")
            print(row[1])
            list = row[1].split(",")
            print(list)
            for model_this in list:
                train_percentage_dict['method'].append(model_this)
        elif line==1:
            print(row[1])
            for char in row[1]:
                if char in " []\'":
                    row[1] = row[1].replace(char,'')
            print("\n\n")
            print(row[1])
            list = row[1].split(",")
            print(list)
            for test_acc in list:
                train_percentage_dict['train_percentage'].append(float(test_acc))
        else:
            print(row[1])
            for char in row[1]:
                if char in " []\'":
                    row[1] = row[1].replace(char,'')
            print("\n\n")
            print(row[1])
            list = row[1].split(",")
            print(list)
            for training_fraction in list:
                train_percentage_dict['best_accuracy'].append(float(training_fraction))
        line+=1
        
print(train_percentage_dict['method'])
print(train_percentage_dict['best_accuracy'])
print(train_percentage_dict['train_percentage'])

print("Restored train_percentage_dict")

train_percentage_df = pd.DataFrame(data=train_percentage_dict)

# plot
sns.set(style="whitegrid")

                
plot = sns.catplot(x="train_percentage", y="best_accuracy", hue="method", data=train_percentage_df, height=6, kind="point", palette="muted", linestyles=[":",":",":", ":", ":"])
plot.despine(left=True)
plot.set_ylabels("Best Accuracy")
plot.set_xlabels("Train Percentage")
plot.savefig("percentage_plot.png")
