---
title: "Project 3 - Rock Typing and Subsurface Mapping"
date: 2021-06-06
tags: [data analytics, data science, classification]
header:
image: "/images/perceptron/percept.jpg"
excerpt: "Classification Modeling in Python using JupyterLab, TechLog, and Petrel"
mathjax: "true"
---

# Overview
### In many subsurface petroleum reservoirs, mobile minerals like anhydrite can occlude pore space.  This can be a major problem, especially when preducted reservoir space is clogged with something other than oil!
#### In this image, anhydrite (the white crystals) occludes almost all of the space!!
![](/images/AnhydriteImages/Anhydrite_Pores.jpg)


## Project Goals
### Use the available subsurface data (petrophysical data and core descriptions) to:
#### 1. Establish an analytical model to account for anhydrite distribution using petrophysical data calibrated to core description.
#### 2. Export that model to adjacent cores, and create regional maps of anhydrite distribution in the subsurface.

![](/images/AnhydriteImages/Anhydrite_Cores.jpg)
#### In these examples we can see (from left to right) an interval of almost pure anhydrite; a muddy interval with anhydrite nodules (about 40% anhydrite); a laminated interval with very little anhydrite.  We want to use data analytics to tell them apart using petropysical data, then use those results to map the occurence of these different volumes of anhydrite elsewhere.


### Step 1. Install Modules and Libraries:
```python
#DS Basics
from sqlalchemy import create_engine
import pymysql
import numpy as np
import pandas as pd
import pandas_profiling
import scipy
import matplotlib.pyplot as plt
import math
import pickle
from math import sqrt
import seaborn as sns
import missingno as msno

#For Log Display:
import lasio
from mpl_toolkits.mplot3d import Axes3D

#SKLearn Stuff
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
```
### Load Data.  Well data usually comes in .las files.  This can be read by Python using lasio. 
```python
#For reading .las files:
las = lasio.read("15-9-19_SR_COMP.LAS")
#Rename columns for log names
data.rename(columns={"CALI":"Caliper","RSHA":"Res (S)","RMED":"Res (M)","RDEP":"Res (Dp)","RHOB":"Density (rhob)", "GR":"GR(raw)","ROP":"ROP", "DTS":"Sonic (ShSl)","DCAL":"Diff. Cal.","DRHO":"Density (corr)","RMIC":"Res (Mic)","ROPA":"ROP (avg)","RXO":"Res (flu)","FORCE_2020_LITHOFACIES_LITHOLOGY":"LITHOLOGY","FORCE_2020_LITHOFACIES_CONFIDENCE":"LITHOLOGY (conf)"}, inplace=True)
data.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>WELL</th>
      <th>DEPTH_MD</th>
      <th>X_LOC</th>
      <th>Y_LOC</th>
      <th>Z_LOC</th>
      <th>GROUP</th>
      <th>FORMATION</th>
      <th>Caliper</th>
      <th>Res (S)</th>
      <th>Res (M)</th>
      <th>...</th>
      <th>ROP</th>
      <th>Sonic (ShSl)</th>
      <th>Diff. Cal.</th>
      <th>Density (corr)</th>
      <th>MUDWEIGHT</th>
      <th>Res (Mic)</th>
      <th>ROP (avg)</th>
      <th>Res (flu)</th>
      <th>LITHOLOGY</th>
      <th>LITHOLOGY (conf)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15/9-13</td>
      <td>494.528</td>
      <td>437641.96875</td>
      <td>6470972.5</td>
      <td>-469.501831</td>
      <td>KHUFF FM.</td>
      <td>NaN</td>
      <td>19.480835</td>
      <td>NaN</td>
      <td>1.611410</td>
      <td>...</td>
      <td>34.636410</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.574928</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>65000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15/9-13</td>
      <td>494.680</td>
      <td>437641.96875</td>
      <td>6470972.5</td>
      <td>-469.653809</td>
      <td>KHUFF FM.</td>
      <td>NaN</td>
      <td>19.468800</td>
      <td>NaN</td>
      <td>1.618070</td>
      <td>...</td>
      <td>34.636410</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.570188</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>65000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15/9-13</td>
      <td>494.832</td>
      <td>437641.96875</td>
      <td>6470972.5</td>
      <td>-469.805786</td>
      <td>KHUFF FM.</td>
      <td>NaN</td>
      <td>19.468800</td>
      <td>NaN</td>
      <td>1.626459</td>
      <td>...</td>
      <td>34.779556</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.574245</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>65000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15/9-13</td>
      <td>494.984</td>
      <td>437641.96875</td>
      <td>6470972.5</td>
      <td>-469.957794</td>
      <td>KHUFF FM.</td>
      <td>NaN</td>
      <td>19.459282</td>
      <td>NaN</td>
      <td>1.621594</td>
      <td>...</td>
      <td>39.965164</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.586315</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>65000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15/9-13</td>
      <td>495.136</td>
      <td>437641.96875</td>
      <td>6470972.5</td>
      <td>-470.109772</td>
      <td>KHUFF FM.</td>
      <td>NaN</td>
      <td>19.453100</td>
      <td>NaN</td>
      <td>1.602679</td>
      <td>...</td>
      <td>57.483765</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.597914</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>65000</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 29 columns</p>
</div>

### Now that we have the data imported we want to be able to display it in a vertical format so that we can compare to physical core description:
```python
fig = plt.subplots(figsize=(7,10))
##Set up the plot axes
ax1 = plt.subplot2grid((1,7), (0,0), rowspan=1, colspan = 1) 
ax2 = plt.subplot2grid((1,7), (0,1), rowspan=1, colspan = 1)
ax3 = plt.subplot2grid((1,7), (0,2), rowspan=1, colspan = 1)
ax4 = plt.subplot2grid((1,7), (0,3), rowspan=1, colspan = 1)
ax5 = plt.subplot2grid((1,7), (0,4), rowspan=1, colspan = 1)
ax6 = plt.subplot2grid((1,7), (0,5), rowspan=1, colspan = 1)
ax7 = plt.subplot2grid((1,7), (0,6), rowspan=1, colspan = 1)
columns = well_nan.columns
axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]
for i, ax in enumerate(axes):
    ax.plot(well_nan.iloc[:,i], well_nan.index, lw=0)
    ax.set_ylim(5000, 0)
    ax.set_xlim(0, 1)
    ax.set_title(columns[i])
    ax.set_facecolor('whitesmoke')
    ax.fill_betweenx(well_nan.index, 0, well_nan.iloc[:,i], facecolor='red')
    ## Remove tick labels from each subplot
    if i > 0:
        plt.setp(ax.get_yticklabels(), visible = False)
    plt.setp(ax.get_xticklabels(), visible = False)
ax1.set_ylabel('Depth', fontsize=14)
```


### However, before we plot all the logs, we want to know which petrophysical measurements will discriminate anhydrite most effciently.  
![](/images/AnhydriteImages/Elan_1.jpg)
### In this case the combined logs that are most effective for descrimination are neutron density, resistivity (deep), sonic (C&S), and neutron porosity



### Now we can isolate the diagnostic component parameters in the dataset:
```python
#Neutron-Density Distribution by Lithology and Well
g = sns.FacetGrid(data, col='WELL', hue='LITH', col_wrap=4)
g.map(sns.scatterplot, 'NPHI', 'RHOB', linewidth=1, size=0.1, marker='+')
g.set(xlim=(-0.15, 1))
g.set(ylim=(3, 1))
g.add_legend()
#Neutron Density Distribution by Lithology and Geological Group
#We can easily swap out the grouping option for the geological GROUP column to view the lithology variation by group.
g = sns.FacetGrid(data, col='GROUP', hue='LITH', col_wrap=4)
g.map(sns.scatterplot, 'NPHI', 'RHOB', linewidth=1, size=0.1, marker='+')
g.set(xlim=(-0.15, 1))
g.set(ylim=(3, 1))
g.add_legend()
```

### Now we have to iterate this process by physically cross-checking the data with the core to make sure what is being predicted is actually there:
![](/images/AnhydriteImages/Prediction_1.jpg)

```python
#Cross apply definitions to all lithologies. 
## Determine number of rows
nrows = int(math.ceil(len(grouped)/4.))
## Group our data
grouped_lith = data.groupby('LITH')
```


### Once we are satisfied that the classification model is performing well, we can visualize the end-members we've defined in TechLog:
![](/images/AnhydriteImages/Property_Response_Space.jpg)
### This shows that we have very little property crossover for our defined petropysical responses that are calibrated to physical core description.


### The next step is to export the model to data from surrounding wells.  Where core is available, we can perform another cross-check to ensure the deployment is accurate before we deploy the model to all the wells in the field:
![](/images/AnhydriteImages/Prediction_2.jpg)



### Once we are satisfied with teh accuracy of the model in other wells, we deploy the model to all wells in the field and create cross-sections:
![](/images/AnhydriteImages/XC.jpg)


### We then load the model data into subsurface mapping software (Petrel) and make maps of predicted anhydrite occurrence at defined stratigraphic intervals:
![](/images/AnhydriteImages/Map.jpg)


