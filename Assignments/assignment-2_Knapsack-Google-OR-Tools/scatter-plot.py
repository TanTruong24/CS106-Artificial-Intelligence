import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib. pyplot as plt


d = {'Time': [0.875, 0.71875, 2082.172, 26176.000025, 1519.8438, 0.5625, 2004.640675, 2555.67193, 2780.468825, 2977.0627, 2481.922, 2567.9377, 3104.85965],
     'Not Optimal': [0, 0, 1, 2, 1, 0, 2, 1, 3, 2, 1, 3, 3]
     }
df = pd.DataFrame(data=d)

df['Not optimal'] = (0, 0, 1, 2, 1, 0, 2, 1, 3, 2, 1, 3, 3)

sns.scatterplot(x="Time", y="Not Optimal", data=df, hue="Not optimal")
plt.show()
