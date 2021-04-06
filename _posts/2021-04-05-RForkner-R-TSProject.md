---
title: "Project 2 - Time Series Analysis in R"
date: 2021-04-05
tags: [data analytics, data science, time series, R]
header:
image: "/images/perceptron/percept.jpg"
excerpt: "Time Series Analysis using RStudio"
mathjax: "true"
---

# Overview
## A home owner has become more aware of their power usage and has decided to track their consumption using 3 power meters in their house.  These meters take a measurement every minute.  By analyzing their power usage, the owner wants to try to predict their consumption in the futute.
# Project Goals
## Use the sub-metered power consumption data to:
### 1. Create a time series of power consumtion for a given period.
### 2. Model the time series and predict future power consumption. 
## Step 1. Install Modules and Libraries.  These libraries will allow for data loading, examination, analysis, visualization, and forecasting:
```r
install.packages("tidyverse")
library(tidyverse)
install.packages("RMySQL")
library(RMySQL)
install.packages("lubridate")
library(lubridate)
install.packages("pastecs")
library(pastecs)
install.packages("ggplot2")
library(ggplot2)
install.packages("ggfortify")
library(ggfortify)
install.packages("plotly")
library(plotly)
install.packages("forecast")
library(forecast)
```
### Step 2. Link to the dataset with MySQL and examine 
```r
con = dbConnect(MySQL(), user='deepAnalytics', password='Sqltask1234!', dbname='dataanalytics2018', host='data-analytics-2018.cbrosir2cswx.us-east-1.rds.amazonaws.com')
# Select the data for each year should be Date, Time and the 3 sub-meter attributes.
yr_2006SELECT <- dbGetQuery(con, "SELECT Date, Time, Sub_metering_1, Sub_metering_2, Sub_metering_3 FROM yr_2006")
dbListFields(con,'yr_2007')
yr_2007SELECT <- dbGetQuery(con, "SELECT Date, Time, Sub_metering_1, Sub_metering_2, Sub_metering_3 FROM yr_2007")
dbListFields(con,'yr_2008')
yr_2008SELECT <- dbGetQuery(con, "SELECT Date, Time, Sub_metering_1, Sub_metering_2, Sub_metering_3 FROM yr_2008")
dbListFields(con,'yr_2009')
yr_2009SELECT <- dbGetQuery(con, "SELECT Date, Time, Sub_metering_1, Sub_metering_2, Sub_metering_3 FROM yr_2009")
dbListFields(con,'yr_2010')
yr_2010SELECT <- dbGetQuery(con, "SELECT Date, Time, Sub_metering_1, Sub_metering_2, Sub_metering_3 FROM yr_2010")
# Combine tables into one dataframe using dplyr
CombinedDF <- bind_rows(yr_2007SELECT, yr_2008SELECT, yr_2009SELECT)
# Since the Date and Time columns are separate they will need to be combined within the dataset in order to convert them to the correct format for time series analysis. 
CombinedDFT <-cbind(CombinedDF,paste(CombinedDF$Date,CombinedDF$Time), stringsAsFactors=FALSE)
```
## Here's an example of the combined dataset.  1,569,894 entries in 6 columns!!!  That's power data from 2006-2010 taken every minute!

<img src="/images/Combined%20DFT.jpg">

### Step 3. We'll now use an R fucntion called 'lubridate' to create "year", "quarter", "month", "week", "day", "hour", and "minute" attributes

```r
CombinedDFT$year <- year(CombinedDFT$DateTime)
CombinedDFT$quarter <- quarter(CombinedDFT$DateTime)
CombinedDFT$month <- month(CombinedDFT$DateTime)
CombinedDFT$week <- week(CombinedDFT$DateTime)
CombinedDFT$day <- day(CombinedDFT$DateTime)
CombinedDFT$hour <- hour(CombinedDFT$DateTime)
CombinedDFT$minute <- minute(CombinedDFT$DateTime)
```
### Step 4. Since data were acquired each minute, the entire dataset will prove difficult to use as a whole, so we'll subset the dataset into time windows
```r
house9Jan2008 <- filter(CombinedDFT, year == 2008 & month == 1 & day == 9)
#' Plot sub-meter 1, 2 and 3 with title, legend and labels - All observations 
plot_ly(house9Jan2008, x = ~house9Jan2008$DateTime, y = ~house9Jan2008$Sub_metering_1, name = 'Kitchen', type = 'scatter', mode = 'lines') %>%
  add_trace(y = ~house9Jan2008$Sub_metering_2, name = 'Laundry Room', mode = 'lines') %>%
  add_trace(y = ~house9Jan2008$Sub_metering_3, name = 'Water Heater & AC', mode = 'lines') %>%
  layout(title = "Power Consumption January 9th, 2008",
         xaxis = list(title = "Time"),
         yaxis = list (title = "Power (watt-hours)"))
```

![](/images/Jan%209%20power%20consumption%20all.jpeg)

### Step 5. Since that uses power measurements taken every minute, we can probably coarsen the data sampling to every 10 minutes and still see the relevant oscillations in power usage
```r
house9Jan2008x10min <- filter(house9Jan2008, year == 2008 & month == 1 & day == 9 & (minute == 0 | minute == 10 | minute == 20 | minute == 30 | minute == 40 | minute == 50))
#' Plot sub-meter 1, 2 and 3 with title, legend and labels - 10 Minute frequency
plot_ly(house9Jan2008x10min, x = ~house9Jan2008x10min$DateTime, y = ~house9Jan2008x10min$Sub_metering_1, name = 'Kitchen', type = 'scatter', mode = 'lines') %>%
  add_trace(y = ~house9Jan2008x10min$Sub_metering_2, name = 'Laundry Room', mode = 'lines') %>%
  add_trace(y = ~house9Jan2008x10min$Sub_metering_3, name = 'Water Heater & AC', mode = 'lines') %>%
  layout(title = "Power Consumption January 9th, 2008",
         xaxis = list(title = "Time"),
         yaxis = list (title = "Power (watt-hours)"))
```
### Here's the same plot with measurements taken every 10 minutes:


![](/images/Jan%209th%20power%20consumption%2010%20min.jpeg)

### Step 6. Now we'll prepare the data for Time Series Analysis.  In this case, to get enough of a sample for forecasting, we'll sample the whole month of January, 2008, with power measurements taken every hour.  We'll use the same code for each submeter.  We'll then plot each sub-meter's readings for the month:
```r
houseJan2008 <- filter(CombinedDFT, year == 2008 & month == 1 & (minute == 0))
houseJan2008$Date<-NULL
houseJan2008$Time<-NULL
View(houseJan2008)
str(houseJan2008)
tshouseJan2008sm_1<-ts(houseJan2008$Sub_metering_1, frequency = 24)
plot(tshouseJan2008sm_1)
```
### Here's a plot of power usage in January, 2008, for each sub meter
![](/images/Jan%202008%20sm1%20kitchen.jpeg)
![](/images/Jan2008%20sm2%20laundry.jpeg)
![](/images/Jan2008%20sm3%20WHAC.jpeg)
### Step 7. In order to perform a forecast on the time series, we will decompose the time data into its component parts: the background trend, repeating seasonality and remaining noise.  This is because when we make a forecast, the model we use for the forecast will focus on the trend and the seasonality in the data.
```r
DCtshouseJan2008sm_1<-decompose(tshouseJan2008sm_1)
autoplot(DCtshouseJan2008sm_1, main = "January, 2008 Sub-meter 1")
```
### Here's a plot of the time series decomposition: 

![](/images/Jan%202008%20sm1%20kitchen%20decomp.jpeg)

### Step 8. Now we will create a time series linear models (tslm) for Jan 2008 for each sub-meter.  There are many models we can apply, but in this case we'll run a simple linear model for the trend and the seasonality.  We'll use these models in the forecast package to forecast usaage for the next day.
```r
fitSM1 <- tslm(tshouseJan2008sm_1 ~ trend + season) 
summary(fitSM1)
forecastfitSM1 <- forecast(fitSM1, h=24, level=c(80,90))
autoplot(forecastfitSM1, colour = 'green', xlab = "Time", ylab = "Watt Hours", main = "January, 2008 Sub-meter 1")
```
### Here's a plot of the power usage forecast for each submeter:  
![](/images/Jan2008%20sm1%20kitchen%20forecast.jpeg)
![](/images/Jan2008%20sm2%20laundry%20forecast.jpeg)
![](/images/Jan2008%20sm3%20WHAC%20forecast.jpeg)
### Each forecast uses the trend and seasonlity to make a prediction of power usage for 24 hours in the future with 80 and 90% confidence bands based on the linear model. 
### Step 9. How will each forecast perform?  We'll need to compare the result to future power usage to find out!



© 2021 GitHub, Inc.
Terms
Privacy
Security
Status
Docs
Contact GitHub
Pricing
API
Training
Blog
About

