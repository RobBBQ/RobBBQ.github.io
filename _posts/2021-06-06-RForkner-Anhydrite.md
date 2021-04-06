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
