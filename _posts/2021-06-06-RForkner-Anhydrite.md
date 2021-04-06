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

```python
#Load Data. 
data = pd.read_csv('train.csv', sep=';')
#Rename columns for log names
data.rename(columns={"CALI":"Caliper","RSHA":"Res (S)","RMED":"Res (M)","RDEP":"Res (Dp)","RHOB":"Density (rhob)", "GR":"GR(raw)","ROP":"ROP", "DTS":"Sonic (ShSl)","DCAL":"Diff. Cal.","DRHO":"Density (corr)","RMIC":"Res (Mic)","ROPA":"ROP (avg)","RXO":"Res (flu)","FORCE_2020_LITHOFACIES_LITHOLOGY":"LITHOLOGY","FORCE_2020_LITHOFACIES_CONFIDENCE":"LITHOLOGY (conf)"}, inplace=True)
data.head()
```
