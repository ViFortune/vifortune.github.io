---
title: Vietnam AQI Forecasting (Air Quality)
layout: post
date: 2025-09-29
categories: [Blog, Tech]
tags: [e2e-ml-pipeline]
author: ltnghia
description: End-to-end data and machine learning pipeline for air quality forecasting in Vietnam.
math: true
image:
  path: /assets/images/air_quality_demo/header.png
  alt: UI of Forecasting Air Quality Web
---

# Vietnam AQI Forecasting

## Introduction

Air pollution is a critical environmental issue affecting public health and urban sustainability. Forecasting air quality indicators helps authorities and citizens better understand pollution patterns and prepare for potential environmental risks.

This project explores whether **air quality indicators can be predicted using historical environmental data** combined with temporal and geographic information.

An **end-to-end machine learning pipeline** was developed including:

1. Data collection from monitoring portals  
2. Data preprocessing  
3. Feature engineering  
4. Dataset construction  
5. Model training (XGBoost)  
6. Web deployment for visualization and forecasting  

Data sources:

-   [https://envisoft.gov.vn](https://envisoft.gov.vn)
-   [https://cem.gov.vn](https://cem.gov.vn)

A demo web application for air quality prediction is available here:

**Demo:** [Air Quality Demo](https://air-quality-forecasting-demo.onrender.com/)
**GitHub:** [Air Quality Forecasting Repository](https://github.com/ViFortune/Air-Quality-Forecasting-Demo/)

> Note: The demo is hosted on Render free tier and may take more than 5 minutes to wake up.

---

# Data Collection

The website **cem.gov.vn** provides hourly air quality measurements from monitoring stations across Vietnam.

Each station contains metadata such as:

```
station_id: "31390908889087377344742439468"
res: { "CO": {...}, "PM-10": {...}, "SO2": {...}, "PM-2-5": {...}, "O3": {...}, "NO2": {...}}
station_name: "Hà Nội: Công viên Nhân Chính - Khuất Duy Tiến (KK)"
```

Important identifiers include:

- `station_id`
- pollutant keys in `res`

Using these identifiers, pollutant measurements can be retrieved.

Due to legal and ethical considerations regarding automated extraction from government portals, the exact scraping implementation is not included.

Example collected dataset:

```
STT,Date,VN_AQI,CO,NO2,O3,PM-10,PM-2-5,SO2
1,06/03/2026,120,21,7,12,76,120,5
2,05/03/2026,74,16,5,5,56,74,7
3,04/03/2026,47,11,5,11,35,47,14
```

Raw data often contains missing or inconsistent attributes:

```
STT,Date,VN_AQI,Benzen,CH4,CO,...
1,05/03/2026,83,-,-,-,...
2,04/03/2026,55,-,-,-,...
```

Therefore, preprocessing is required before training models.

---

# Data Preprocessing

## Removing unnecessary columns

Across monitoring stations, the most consistently available pollutants are:

- PM2.5  
- PM10  
- CO  
- SO2  

Columns with more than **90% missing values** are removed.  
The `STT` column is also dropped.

Cleaned dataset:

```
Date,VN_AQI,CO,PM-10,PM-2-5,SO2
06/03/2026,120,21,76,120,5
05/03/2026,74,16,56,74,7
04/03/2026,47,11,35,47,14
```

---

## Handling missing values

Environmental datasets often contain missing values due to sensor downtime or communication issues.

Missing measurements are filled using **bidirectional linear interpolation**.

```python
def fill_missing(csv_obj: pd.DataFrame):

    cols = ['VN_AQI','CO','PM-10','PM-2-5','SO2',
            'mon','tu','wed','thu','fri','sat','sun',
            'north','middle','south',
            'spring','summer','autumn','winter','dry','rain']

    csv_obj[cols] = csv_obj[cols].apply(pd.to_numeric).astype(np.float64)

    csv_obj['CO'] = csv_obj['CO'].interpolate(method='linear', limit_direction='both')
    csv_obj['PM-10'] = csv_obj['PM-10'].interpolate(method='linear', limit_direction='both')
    csv_obj['PM-2-5'] = csv_obj['PM-2-5'].interpolate(method='linear', limit_direction='both')
    csv_obj['SO2'] = csv_obj['SO2'].interpolate(method='linear', limit_direction='both')

    return csv_obj
```

---

# Exploratory Data Analysis

### Distribution

Pollutant values roughly follow a bell-shaped distribution, though not strictly Gaussian. This is typical for environmental data influenced by many stochastic factors.

### Temporal behavior

Time-series visualization shows that pollutants such as **PM2.5, PM10, CO and SO2** often increase together during certain periods.

For example, from **September 2024 to May 2025**, these indicators tend to rise simultaneously, suggesting possible seasonal effects.

However, longer observation periods are required for statistical confirmation.

---

# Feature Engineering

Additional features were created to capture temporal and geographic patterns.

## Day of Week

Air quality may change between weekdays and weekends due to differences in traffic and industrial activity.

One-hot encoding is used:

```
mon, tu, wed, thu, fri, sat, sun
```

## Geographic Region

Vietnam is divided into three regions:

- North
- Central
- South

These regions have distinct climates and pollution patterns.

## Seasonal Indicators

Seasonal variables are derived from region and month:

```
spring, summer, autumn, winter, dry, rain
```

Example dataset after feature engineering:

```
Date,Day_Of_Week,VN_AQI,CO,PM-10,PM-2-5,SO2,mon,tu,wed,thu,fri,sat,sun,north,middle,south,spring,summer,autumn,winter,dry,rain
2026-03-06,4,120,21,76,120,5,0,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0
```

---

# Data Storage with MongoDB

After preprocessing, data is stored in **MongoDB Atlas**.

MongoDB was chosen because:

- JSON-like document structure
- Good compatibility with Python data pipelines
- Easy horizontal scalability

Example connection:

```python
from pymongo import MongoClient
import os

MONGO_URI = os.getenv("MONGO_URI")

def get_mongo_client():
    client = MongoClient(host=MONGO_URI)
    return client
```

Configuration via environment variables:

```python
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
```

---

# Converting DataFrame to MongoDB Documents

Pandas DataFrames are converted into JSON-like documents before insertion.

```python
client = get_mongo_client()

db = client[DB_NAME]
collection = db[COLLECTION_NAME]

records = df.to_dict(orient='records')
collection.insert_many(records)
```

Example document:

```json
{
  "Date": "2026-03-06",
  "VN_AQI": 120,
  "CO": 21,
  "PM-10": 76,
  "PM-2-5": 120,
  "SO2": 5
}
```

Each record becomes a MongoDB document.

---

# Dataset Construction

The time series data is transformed into a supervised learning problem using a **sliding window**.

For each training sample:

- **Previous 7 days** → input features  
- **Next day** → prediction target

This captures short-term temporal dependencies while keeping the model simple.

---

# Model Training

The task is formulated as a **regression problem**.

Separate models are trained for:

- CO
- PM10
- PM2.5
- SO2

Algorithm: **XGBoost**

```python
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.05,
    n_jobs=-1,
    random_state=42
)
```

Dataset split:

- 80% training  
- 10% validation  
- 10% testing  

Chronological order is preserved to avoid data leakage.

Trained models are stored using `joblib`.

---

# Data Pipeline & Storage Integration

MongoDB also supports pipeline state management.

Pipeline routine:

1. **Crawl & Sync** – collect latest monitoring data  
2. **State Management** – store `last_run_time` in a separate collection  
3. **Preprocessing** – clean data and generate model inputs  

This design allows the system to resume processing reliably.

---

# Modernized Web Architecture

The newer version replaces server-side plotting with a **JSON API architecture**.

## Backend (Flask API)

Instead of generating static plots, Flask provides an API returning compact JSON:

- **Dates** – timeline sequence  
- **Values** – pollutant indicators  
- **Metadata** – indices separating real and predicted values  

## Client-side Visualization (Plotly.js)

Charts are rendered directly in the browser using Plotly.

Advantages:

- Interactive UI
- Stateless deployment
- Responsive charts
- Real-time updates via AJAX

---

# Web Application and Deployment

The web interface is built using **Flask**.

Workflow:

1. Retrieve latest monitoring data  
2. Preprocess and generate features  
3. Construct model inputs  
4. Forecast next **7 days**

Predictions are generated **recursively**:
prediction of day `t+1` becomes input for predicting `t+2`.

A scheduled job runs daily at **05:00** using:

- `cron`
- `apscheduler`

Deployment:

- Render (web hosting)
- GitHub integration

---

# Limitations and Future Work

This project is an initial exploration of air quality forecasting.

Possible improvements:

- Automated model retraining  
- Integration of meteorological data  
- Additional monitoring stations  
- More advanced evaluation metrics  

Future models may include:

- LSTM  
- Temporal CNN  
- Transformer-based forecasting models  

Despite its simplicity, the system demonstrates how an **end-to-end machine learning pipeline** can be built and deployed for environmental monitoring.