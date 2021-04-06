---
title: "Project 2 - Time Series Analysis in R"
date: 2021-04-05
tags: [data analytics, data science, time series, R]
header:
image: "/images/perceptron/percept.jpg"
excerpt: "Data Science, Messy Data, Data Analytics"
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
### Step 2. Link to the dataset and examine 
```r
#' Create a database connection: 
con = dbConnect(MySQL(), user='deepAnalytics', password='Sqltask1234!', dbname='dataanalytics2018', host='data-analytics-2018.cbrosir2cswx.us-east-1.rds.amazonaws.com')
#' List the tables contained in the database 
dbListTables(con)
#' Select the data for each year should be Date, Time and the 3 sub-meter attributes.
yr_2006SELECT <- dbGetQuery(con, "SELECT Date, Time, Sub_metering_1, Sub_metering_2, Sub_metering_3 FROM yr_2006")
dbListFields(con,'yr_2007')
yr_2007SELECT <- dbGetQuery(con, "SELECT Date, Time, Sub_metering_1, Sub_metering_2, Sub_metering_3 FROM yr_2007")
dbListFields(con,'yr_2008')
yr_2008SELECT <- dbGetQuery(con, "SELECT Date, Time, Sub_metering_1, Sub_metering_2, Sub_metering_3 FROM yr_2008")
dbListFields(con,'yr_2009')
yr_2009SELECT <- dbGetQuery(con, "SELECT Date, Time, Sub_metering_1, Sub_metering_2, Sub_metering_3 FROM yr_2009")
dbListFields(con,'yr_2010')
yr_2010SELECT <- dbGetQuery(con, "SELECT Date, Time, Sub_metering_1, Sub_metering_2, Sub_metering_3 FROM yr_2010")
#' Combine tables into one dataframe using dplyr
CombinedDF <- bind_rows(yr_2007SELECT, yr_2008SELECT, yr_2009SELECT)
#' Use str(), summary(), head() and tail() with the combined data frame to check dates.
str(CombinedDF)
summary(CombinedDF)
head(CombinedDF) 
tail(CombinedDF)
#' Since the Date and Time columns are separate they will need to be combined within the dataset in order to convert them to the correct format for time series analysis. 
CombinedDFT <-cbind(CombinedDF,paste(CombinedDF$Date,CombinedDF$Time), stringsAsFactors=FALSE)
#' Give the new attribute in the 6th column a header name 
colnames(CombinedDFT)[6] <-"DateTime"
#' Move the DateTime attribute within the dataset
CombinedDFT <- CombinedDFT[,c(ncol(CombinedDFT), 1:(ncol(CombinedDFT)-1))]
head(CombinedDFT)
#' We will now want to convert the new DateTime attribute to the POSIXlt class that stores date/time values as a list of components (hour, min, sec, mon, etc.) making it easy to extract these parts.
CombinedDFT$DateTime <- as.POSIXct(CombinedDFT$DateTime, "%Y/%m/%d %H:%M:%S")
#' Add the time zone
attr(CombinedDFT$DateTime, "tzone") <- "Europe/Paris"
#' Inspect the data types
str(CombinedDFT)
#' Create "year", "quarter", "month", "week", "day", "hour", and "minute" attributes with lubridate
CombinedDFT$year <- year(CombinedDFT$DateTime)
CombinedDFT$quarter <- quarter(CombinedDFT$DateTime)
CombinedDFT$month <- month(CombinedDFT$DateTime)
CombinedDFT$week <- week(CombinedDFT$DateTime)
CombinedDFT$day <- day(CombinedDFT$DateTime)
CombinedDFT$hour <- hour(CombinedDFT$DateTime)
CombinedDFT$minute <- minute(CombinedDFT$DateTime)
#' Calculate the mean, mode, standard deviation, quartiles & characterization of the distribution
summary(CombinedDFT)
#' Calculate descriptive statistics for sub-meters
stat.desc(CombinedDFT$Sub_metering_1)
stat.desc(CombinedDFT$Sub_metering_2)
stat.desc(CombinedDFT$Sub_metering_3)
#' Since data were acquired each minute, we'll subset the dataset to view time windows:
house9Jan2008 <- filter(CombinedDFT, year == 2008 & month == 1 & day == 9)
#' Plot sub-meter 1, 2 and 3 with title, legend and labels - All observations 
plot_ly(house9Jan2008, x = ~house9Jan2008$DateTime, y = ~house9Jan2008$Sub_metering_1, name = 'Kitchen', type = 'scatter', mode = 'lines') %>%
  add_trace(y = ~house9Jan2008$Sub_metering_2, name = 'Laundry Room', mode = 'lines') %>%
  add_trace(y = ~house9Jan2008$Sub_metering_3, name = 'Water Heater & AC', mode = 'lines') %>%
  layout(title = "Power Consumption January 9th, 2008",
         xaxis = list(title = "Time"),
         yaxis = list (title = "Power (watt-hours)"))
#' We can probably coarsen the dataset somewhat: Subset the 9th day of January 2008 - 10 Minute frequency
house9Jan2008x10min <- filter(house9Jan2008, year == 2008 & month == 1 & day == 9 & (minute == 0 | minute == 10 | minute == 20 | minute == 30 | minute == 40 | minute == 50))
#' Plot sub-meter 1, 2 and 3 with title, legend and labels - 10 Minute frequency
plot_ly(house9Jan2008x10min, x = ~house9Jan2008x10min$DateTime, y = ~house9Jan2008x10min$Sub_metering_1, name = 'Kitchen', type = 'scatter', mode = 'lines') %>%
  add_trace(y = ~house9Jan2008x10min$Sub_metering_2, name = 'Laundry Room', mode = 'lines') %>%
  add_trace(y = ~house9Jan2008x10min$Sub_metering_3, name = 'Water Heater & AC', mode = 'lines') %>%
  layout(title = "Power Consumption January 9th, 2008",
         xaxis = list(title = "Time"),
         yaxis = list (title = "Power (watt-hours)"))
#' Now we'll prepare the data for Time Series Analysis
#' First, we'll Sample the month of January 2008, with a sample taken every hour.  We then null out time to let frequency get autopicked
houseJan2008 <- filter(CombinedDFT, year == 2008 & month == 1 & (minute == 0))
houseJan2008$Date<-NULL
houseJan2008$Time<-NULL
View(houseJan2008)
str(houseJan2008)
#' Subset to sub_meter 1; Frequency = 31 days * 24 hours.  Note: we know the main frequency is 24 hours by cross checking with periodogram.
tshouseJan2008sm_1<-ts(houseJan2008$Sub_metering_1, frequency = 24)
plot(tshouseJan2008sm_1)
#' Decompose
DCtshouseJan2008sm_1<-decompose(tshouseJan2008sm_1)
autoplot(DCtshouseJan2008sm_1, main = "January, 2008 Sub-meter 1")
#' Plot sub-meter 1 with plot.ts
plot.ts(tshouseJan2008sm_1)
#' Plot sub-meter 1 with autoplot - add labels, color
autoplot(tshouseJan2008sm_1, colour = 'green', xlab = "Time", ylab = "Watt Hours", main = "January, 2008 Sub-meter 1")
#' Subset to sub_meter 2; Frequency = 31 days * 24 hours
tshouseJan2008sm_2<-ts(houseJan2008$Sub_metering_2, frequency = 24)
plot(tshouseJan2008sm_2) 
#' Decompose
DCtshouseJan2008sm_2<-decompose(tshouseJan2008sm_2)
autoplot(DCtshouseJan2008sm_2, main = "January, 2008 Sub-meter 2")
#' Plot sub-meter 2 with plot.ts
plot.ts(tshouseJan2008sm_2)
#' Plot sub-meter 2 with autoplot - add labels, color
autoplot(tshouseJan2008sm_2, colour = 'blue', xlab = "Time", ylab = "Watt Hours", main = "January, 2008 Sub-meter 2")
#' Subset to sub_meter 3; Frequency = 31 days * 24 hours
tshouseJan2008sm_3<-ts(houseJan2008$Sub_metering_3, frequency = 24)
plot(tshouseJan2008sm_3) 
#' Decompose
DCtshouseJan2008sm_3<-decompose(tshouseJan2008sm_3)
autoplot(DCtshouseJan2008sm_3, main = "January, 2008 Sub-meter 3")
#' Plot sub-meter 3 with plot.ts
plot.ts(tshouseJan2008sm_3)
#' Plot sub-meter 3 with autoplot - add labels, color
autoplot(tshouseJan2008sm_3, colour = 'red', xlab = "Time", ylab = "Watt Hours", main = "January, 2008 Sub-meter 3")
#' Now we will create three different time series linear models (tslm) for Jan 2008 for each sub-meter.  We'll use these models in the forecast package to forecast usaage for the next day.
fitSM1 <- tslm(tshouseJan2008sm_1 ~ trend + season) 
summary(fitSM1)
#' Create the forecast for sub-meter 1. Forecast ahead 1 day (h=24), with confidence levels 80 and 90 
forecastfitSM1 <- forecast(fitSM1, h=24, level=c(80,90))
#' Plot the forecast for sub-meter 1. 
autoplot(forecastfitSM1, colour = 'green', xlab = "Time", ylab = "Watt Hours", main = "January, 2008 Sub-meter 1")
#' Apply time series linear regression to the sub-meter 2 ts 
fitSM2 <- tslm(tshouseJan2008sm_2 ~ trend + season) 
summary(fitSM2)
#' Create the forecast for sub-meter 2.  Forecast ahead 1 day (h=24), with confidence levels 80 and 90 
forecastfitSM2 <- forecast(fitSM2, h=24, level=c(80,90))
#' Plot the forecast for sub-meter 2. 
autoplot(forecastfitSM2, colour = 'blue', xlab = "Time", ylab = "Watt Hours", main = "January, 2008 Sub-meter 2")
#' Apply time series linear regression to the sub-meter 3 ts 
fitSM3 <- tslm(tshouseJan2008sm_3 ~ trend + season) 
summary(fitSM3)
#' Create the forecast for sub-meter 3.  Forecast ahead 1 day (h=24), with confidence levels 80 and 90 
forecastfitSM3 <- forecast(fitSM3, h=24, level=c(80,90))
#' Plot the forecast for sub-meter 3. 
autoplot(forecastfitSM3, colour = 'red', xlab = "Time", ylab = "Watt Hours", main = "January, 2008 Sub-meter 3")

```


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

