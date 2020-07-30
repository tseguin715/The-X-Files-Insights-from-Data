#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
pd.options.display.max_rows = 999
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats

import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.statespace.sarimax import SARIMAX
plt.style.use('ggplot')

df=pd.read_csv('table_base.csv',index_col='No. overall')
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

import ast
x, y = np.mgrid[-2:df.shape[0]+2:.05, 0:12:.05]
pos = np.dstack((x, y))

rvs=[]
for i in range(df.shape[0]):
    ep=ast.literal_eval(df['ratings'].iloc[i])
    un,cnts = np.unique(ep, return_counts=True)
    cnts=cnts/max(cnts)
    for j in range(len(un)):
        rvs.append(multivariate_normal([i+1, un[j]],[[36, 0], [0, 1]]).pdf(pos)*cnts[j])

fig2 = plt.figure(figsize=(10,4))
ax2 = fig2.add_subplot(111)

ax2.contourf(x, y, sum(rvs),levels=50,cmap=cm.coolwarm)

z=np.array(sum(rvs))
inds=np.argmax(z, axis=1)

from scipy.interpolate import interp1d

x_=[x[i][inds[i]] for i in range(len(inds))]
y_=[y[i][inds[i]] for i in range(len(inds))]
z_=[z[i][inds[i]] for i in range(len(inds))]

x_2=[]
y_2=[]
z_2=[]

x_2.append(x_[0])
y_2.append(y_[0])
z_2.append(z_[0])

for i in range(len(x_)-1):
    if abs(y_[i+1]-y_[i])>0.01:
        x_2.append(x_[i])
        y_2.append(y_[i])
        z_2.append(z_[i])
    else:
        continue

x_2.append(x_[-1])
y_2.append(y_[-1])
z_2.append(z_[-1])

f1 = interp1d(x_2, y_2, kind='cubic')
f2 = interp1d(x_2, z_2, kind='cubic')
xnew = np.linspace(x_2[0], x_2[-1], num=400, endpoint=True)

nums=df.reset_index()[['Season','No. overall']].groupby('Season').min()
nums['No. overall']-=1

ax2.set_xticks(list(nums['No. overall']))
ax2.set_xticklabels(list(nums.index))
ax2.set_xlabel('Season')
ax2.set_ylabel('Rating')
ax2.set_title('Ratings probability map')

fig2.tight_layout()

plt.savefig('img/surf1.png',dpi=1000,bbox_inches = 'tight')

fig2 = plt.figure(figsize=(10,4))
ax2 = fig2.add_subplot(111)
ax2.contourf(x, y, sum(rvs),levels=50,cmap=cm.coolwarm)

ax2.plot(xnew,f1(xnew),color='k',linestyle='dashed')

ax2.set_ylim(6,10.5)

ax2.set_xticks(list(nums['No. overall']))
ax2.set_xticklabels(list(nums.index))
ax2.set_xlabel('Season')
ax2.set_ylabel('Rating')
ax2.set_title('Ratings probability map - the "Ratings Ridge"')

fig2.tight_layout()

plt.savefig('img/surf2.png',dpi=1000,bbox_inches = 'tight')
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import ast

nums=df.reset_index()[['Season','No. overall']].groupby('Season').min()
nums['No. overall']-=1

fig = plt.figure(figsize=(10,6))
ax = Axes3D(fig)

x, y = np.mgrid[-2:df.shape[0]+2:.1, 0:12:.06]
pos = np.dstack((x, y))

rvs=[]
for i in range(df.shape[0]):
    ep=ast.literal_eval(df['ratings'].iloc[i])
    un,cnts = np.unique(ep, return_counts=True)
    cnts=cnts/max(cnts)
    for j in range(len(un)):
        rvs.append(multivariate_normal([i+1, un[j]],[[36, 0], [0, 1]]).pdf(pos)*cnts[j])

ax.scatter(x, y, sum(rvs).flatten(),c= sum(rvs).flatten(),cmap=cm.coolwarm)

ax.set_xticks(list(nums['No. overall']))
ax.set_xticklabels(list(nums.index))
ax.set_xlabel('Season',labelpad=10)
ax.set_ylabel('Rating',labelpad=10)
ax.set_title('Ratings probability surface')
fig.tight_layout()
plt.savefig('img/surf3.png',dpi=1000,bbox_inches = 'tight')

fig = plt.figure(figsize=(10,6))
ax = Axes3D(fig)

ax.scatter(x, y, sum(rvs).flatten(),c= sum(rvs).flatten(),cmap=cm.coolwarm)

ax.scatter(xnew[25:],f1(xnew)[25:],f2(xnew)[25:]+0.02,c='k',s=1,label='Ridge')

ax.set_xticks(list(nums['No. overall']))
ax.set_xticklabels(list(nums.index))
ax.set_xlabel('Season',labelpad=10)
ax.set_ylabel('Rating',labelpad=10)
ax.set_title('Ratings probability surface - the "Ratings Ridge"')

ax.view_init(70, -40)
fig.tight_layout()
plt.savefig('img/surf4.png',dpi=1000,bbox_inches = 'tight')

# %%
