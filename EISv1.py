import cleandata6test
from PyEIS import *
import seaborn as sns

from pandas6 import load_df
dfs = load_df()

x = dfs[0]

ex1 = EIS_exp(data = x)

