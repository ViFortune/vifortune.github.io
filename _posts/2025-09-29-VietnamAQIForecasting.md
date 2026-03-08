---
title: Vietnam AQI Forecasting (Air Quality)
layout: post
data: 2025-09-29
categories: [Blog, Tech]
tags: [e2e-ml-pipeline]
author: ltnghia
description: This post discusses about the End-to-End Data and Machine Learning pipeline and Web deployment to visualize the results.
math: true
image:
  path: /assets/images/air_quality_demo/header.png
  alt: UI of Forecasting Air Quality Web
---

# Vietnam AQI Forecasting

## Introduction

Air pollution is a major environmental issue that directly affects
public health and urban sustainability. Forecasting air quality
indicators allows authorities and citizens to better prepare for
pollution events and understand environmental patterns over time.

This project explores whether air quality indicators at a specific
location can be predicted from:

-   Historical pollutant measurements
-   Temporal features (day of week)
-   Geographic information
-   Seasonal characteristics

To investigate this hypothesis, an **end‑to‑end data and machine
learning pipeline** was developed.\
The system includes the following stages:

1.  Data collection from official environmental monitoring portals
2.  Data cleaning and preprocessing
3.  Feature engineering
4.  Dataset construction for supervised learning
5.  Model training using XGBoost
6.  Web deployment for visualization and forecasting

The main data sources used in this project are:

-   [https://envisoft.gov.vn](https://envisoft.gov.vn)
-   [https://cem.gov.vn](https://cem.gov.vn)

A demo web application for air quality prediction is available here:

**Demo:** [Air Quality Forecasting Demo](http://air-quality-forecasting-demo.onrender.com)
**GitHub:** [Air Quality Forecasting Repository](https://github.com/ViFortune/Air-Quality-Forecasting-Demo)

Note: the web application is deployed on the free tier of Render,
therefore it may enter sleep mode after inactivity and require around
1--2 minutes to restart.

------------------------------------------------------------------------

## Data Collection

The website **cem.gov.vn** provides hourly air quality information from
monitoring stations across Vietnam. By inspecting the structure of the
web interface, it is possible to identify metadata associated with each
station:

    station_id: "31390908889087377344742439468"
    res: { "CO": {...}, "PM-10": {...}, "SO2": {...}, "PM-2-5": {...}, "O3": {...}, "NO2": {...}}
    station_name: "Hà Nội: Công viên Nhân Chính - Khuất Duy Tiến (KK)"

The most important elements are:

-   `station_id`
-   pollutant keys contained in `res`

These identifiers can be used to retrieve pollutant measurements from
the monitoring system.

Due to legal and ethical considerations regarding automated data
extraction from government websites, the exact implementation details of
the scraping process are not included in this article.

For each monitoring station, the collected data is stored in a CSV file
with daily AQI indicators.

Example:

    STT,Date,VN_AQI,CO,NO2,O3,PM-10,PM-2-5,SO2
    1,06/03/2026,120,21,7,12,76,120,5
    2,05/03/2026,74,16,5,5,56,74,7
    3,04/03/2026,47,11,5,11,35,47,14
    4,03/03/2026,41,5,2,8,32,41,2
    5,02/03/2026,51,15,4,13,41,51,8

However, raw data obtained from monitoring portals often contains
missing or inconsistent fields.

Example of a raw dataset with many missing attributes:

    STT,Date,VN_AQI,Benzen,CH4,CO,Compass,...
    1,05/03/2026,83,-,-,-,-,-
    2,04/03/2026,55,-,-,-,-,-

Therefore, a preprocessing pipeline is required before training any
machine learning model.

------------------------------------------------------------------------

## Data Preprocessing

### Removing unnecessary columns

After examining the monitoring data across stations, the most
consistently available pollutants are:

-   PM2.5
-   PM10
-   CO
-   SO2

Other pollutants such as NO2 or O3 appear less frequently and often
contain a high proportion of missing values.

Columns with more than **90% missing values** are removed.\
The column `STT` is also discarded because it only represents row
indices.

The cleaned dataset therefore becomes:

    Date,VN_AQI,CO,PM-10,PM-2-5,SO2
    06/03/2026,120,21,76,120,5
    05/03/2026,74,16,56,74,7
    04/03/2026,47,11,35,47,14
    03/03/2026,41,5,32,41,2

This tabular structure is suitable for downstream analysis and modeling.

------------------------------------------------------------------------

### Handling missing values

Environmental datasets often contain missing measurements due to sensor
downtime or communication issues.

To handle this issue, missing values are filled using **linear
interpolation** from both directions of the time series.

``` python
def fill_missing(csv_obj: pd.DataFrame):

    cols = ['VN_AQI','CO','PM-10','PM-2-5','SO2',
            'mon','tu','wed','thu','fri','sat','sun',
            'north','middle','south',
            'spring','summer','autumn','winter','dry','rain'
            ]

    csv_obj[cols] = csv_obj[cols].apply(pd.to_numeric).astype(np.float64)

    csv_obj['CO'] = csv_obj['CO'].interpolate(method='linear', limit_direction='both')
    csv_obj['PM-10'] = csv_obj['PM-10'].interpolate(method='linear', limit_direction='both')
    csv_obj['PM-2-5'] = csv_obj['PM-2-5'].interpolate(method='linear', limit_direction='both')
    csv_obj['SO2'] = csv_obj['SO2'].interpolate(method='linear', limit_direction='both')

    return csv_obj
```

------------------------------------------------------------------------

## Exploratory Data Analysis

### Distribution of pollutant indicators

The histogram distribution of the pollutant values roughly resembles a
bell‑shaped curve, although it does not strictly follow a normal
distribution. This is expected for real‑world environmental data where
pollutant concentrations are influenced by multiple stochastic factors.

### Temporal behavior

Time‑series visualization shows that pollutant indicators often increase
simultaneously during certain periods of the year.

For example, from **September 2024 to May 2025**, concentrations of
PM2.5, PM10, CO and SO2 appear to increase together. This suggests
potential seasonal or meteorological influences.

However, a longer time span of data would be required to confirm these
patterns statistically.

------------------------------------------------------------------------

## Feature Engineering

Based on the initial hypothesis, several additional features were
engineered to capture temporal and geographic patterns that may
influence air quality.

#### Day of Week

The `Day_Of_Week` feature is derived from the `Date` column.\
The hypothesis is that air quality may improve during weekends when
industrial activity and transportation decrease.

To encode this information for machine learning models, one‑hot encoding
is applied:

    mon, tu, wed, thu, fri, sat, sun

#### Geographic Region

Vietnam is divided into three broad regions:

-   North
-   Central
-   South

These regions have different climates and seasonal patterns which may
influence pollutant dispersion.

#### Seasonal Indicators

Seasonal variables are also derived from geographic region and month:

    spring, summer, autumn, winter, dry, rain

Example dataset after feature engineering:

    Date,Day_Of_Week,VN_AQI,CO,PM-10,PM-2-5,SO2,mon,tu,wed,thu,fri,sat,sun,north,middle,south,spring,summer,autumn,winter,dry,rain
    2026-03-06,4,120,21,76,120,5,0,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0

------------------------------------------------------------------------

## Dataset Construction

To convert the time series data into a supervised learning problem, a
**sliding window approach** is used.

For each sample:

-   The previous **7 days** of pollutant measurements are used as input
    features.
-   The **next day** pollutant values become the prediction targets.

This design captures short‑term temporal dependencies while keeping the
model structure simple.

The implementation is shown below.

``` python
def prepare_data(dir, dst_pt, lagged_number=7):

    cols = ['Date','Day_Of_Week','CO','PM-10','PM-2-5','SO2','VN_AQI',
            'mon','tu','wed','thu','fri','sat','sun',
            'north','middle','south',
            'spring','summer','autumn','winter','dry','rain']

    data_dict = {
        "X":[],
        "CO":[],
        "PM-10":[],
        "PM-2-5":[],
        "SO2":[]
    }
```

Each sample therefore contains:

-   Lagged pollutant values (7 days)
-   Contextual features of the target day

------------------------------------------------------------------------

## Model Training

The forecasting task is formulated as a **regression problem**.

Separate models are trained for each pollutant:

-   CO
-   PM10
-   PM2.5
-   SO2

The algorithm used is **XGBoost**, a gradient boosting method known for
strong performance on tabular datasets.

``` python
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.05,
    n_jobs=-1,
    random_state=42
)
```

The dataset is split into:

-   80% training
-   10% validation
-   10% testing

The most recent observations are reserved for validation and testing to
preserve the chronological order of the time series.

Trained models are saved using `joblib` so they can later be loaded by
the web application.

------------------------------------------------------------------------

## Web Application and Deployment

A simple web interface was implemented using **Flask**.

The system performs the following steps automatically:

1.  Retrieve the latest monitoring data
2.  Run preprocessing and feature engineering
3.  Construct model inputs
4.  Generate predictions for the next 7 days

Predictions are generated recursively:\
the prediction of day `t+1` becomes part of the input for predicting day
`t+2`.

To keep the system up‑to‑date, a scheduled job runs every morning at
**05:00** using `cron` and `apscheduler` to refresh the dataset and
update predictions.

The web service is deployed on **Render** and connected directly to the
GitHub repository.

------------------------------------------------------------------------

## Limitations and Future Work

This project represents a preliminary exploration of air quality
forecasting.

Several improvements could be explored in future work:

-   Automated periodic retraining as new data becomes available

-   Integration of meteorological variables (temperature, humidity, wind
    speed)

-   Inclusion of additional monitoring stations

-   Evaluation using more comprehensive metrics

-   Exploration of advanced time‑series models such as

    -   LSTM
    -   Temporal CNN
    -   Transformer‑based forecasting models

Despite its simplicity, the current system demonstrates how an
end‑to‑end machine learning pipeline can be built for environmental
monitoring and deployed as an accessible web application.
