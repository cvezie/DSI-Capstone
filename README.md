# Reducing the Drive Alone Rate in Seattle, WA

## Problem Statement
How can businesses play a part in reducing the number of employees that commute by driving alone each day? Taking cars off the road during commuting hours not only reduces traffic but also drastically reduces the environmental impact of that city. With climate change a frequent topic in the news and global city populations on the rise, Seattle, and other cities across the country, are creating policies that address these concerns. 

One specific program is the 'Commute Trip Reduction (CTR)' Program run by the *City of Seattle* in partnership with *Commute Seattle*. The program first began in 1991. The law states that any employer with more than 100 employees, who report to work at a single site between 6-9am must participate in the program by developing programs to help employees reduce their drive alone commute trips and conduct commuter surveys biennially.  
Read more here: http://www.seattle.gov/waytogo/ctr_req.htm

## Data and Predictors
For my project, I use data collected through the CTR program to evaluate the benefits offered to employees relating to commuting and to identify which benefits impact the Drive-Alone Rate most (either positively or negatively) in Seattle, WA. 

I formulated this question as a regression problem where the feature, `Alone_Share`, is the target. `Alone_Share` includes single occupancy vehicle drivers but does not include solo motorcycle drivers.

$$Alone Share = \frac{Weekly Drive Alone Trips}{Total Weekly Trips}$$

#### `program_report_data`:
This table contains information regarding the benefits offered to employees by individual employers. Additional data includes how much money a company spends on those benefits, how and how often information about commuting program is distributed, and number of employees taking advantage of the individual subsidies offered to them.

There are 51 features that represent different types of benefits offered and facilities available to employees.

#### `worksites_in_goal`:
This table contains survey responses about how employees commute to work throughout the course of a week. It also contains the drive-alone-rate goal for each company, percentage of surveys returned, and the year in which the survey was collected. This table contains information beyond the bounds of the City of Seattle, but I will only use Seattle data for the purposes of this project. 

There are 1182 rows containing data from companies in Seattle, WA but there are only 328 unique CTR_ID codes. 

## Benchmark Model
* LassoCV: 0.959261
* BayesianRidge: 0.967083
* DecisionTree: 0.914122
* KNN: 0.799720

## Project Roadmap
1. Load data from csv files
    * I will not upload this notebook to GitHub as it contains proprietary information
1. Merge datasets
1. Clean data
    * Edit column headers approrpriately
    * Remove obviously meaningless columns (that contain either all the same values or no values at all)
    * Anonymize data
1. Exploratory Data Analysis
    * Use distplots for numerical features
    * Use boxplots for categorical features
    * Show correlation between features using a heatmap
    * Show relationship between target (alone share) and numeric features
1. Preprocessing
    * Deskew and scale numeric features
    * Remove outliers using Tukey's Method
    * Encode categorical features
1. PCA
    * On numerical features
    * Cluster analysis on PCA components
    * Append PCA data to normalized DataFrame
1. Modeling
    * LassoCV: 0.971045
    * BayesianRidgeRegressor: 0.964379
    * DecisionTree: 0.903892
    * KNN: 0.664433
1. Analysis
    * Features likely to decrease drive-alone share
        * Bus Share
        * Total Employees
        * Carpool Share
        * Telecommute Share
        * Last time distributed commute program info to employees - 08/30/2016
               
    * Features likely to increase drive-alone share
        * Principal Component 2
        * Total Annual Greenhouse Gas per Employee in Metric Tons
        * Aggregate Pounds of Greenhouse Gas 
        * Additional Benefits - None
        * Vanshare/Carpool Subsidy - 50-59%


## Stretch Goals
1. Cost analysis of programs and alone-share
1. Use Django to enable front-end interaction with the model
1. Analysis over time
1. GIS to find areas of low/high alone-share