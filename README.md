# Classifying Life Expectancy

**Group Partners: Filis, Manisha, Pablo**

Module 5 project

## Table of Contents
* [The Question](#question)
* [Conclusion](#conclusion)
* [The Data](#data)
* [Data Cleaning](#cleaning)
* [Classifier](#classify)
* [Definitions](#glossary)
* [Columns](#columns)

---

## <a name="question"></a>The Question

compare population trends between developing / developed -- which features contribute towards life expectancy -- correlated with population trends (too many old people or too many young people . . .). What's happening?? 

<!-- Put map here -->
![World Map](./images/world_map.png)

We compute the median life expectancy from all countries across all years. If a country's life expectancy is lower than the median, it has "low" life expectancy. Higher life expectancy is considered "high". This way, we transform the continuous life expectancy values into a binary class.


## <a name="conclusion"></a>Conclusion

We found that schooling is the most important feature in classifying whether a country has low life expectancy or not. **WHAT ELSE**



## <a name="data"></a>The Data

We use data from the World Health Organization found on Kaggle ([link here](https://www.kaggle.com/kumarajarshi/life-expectancy-who)). It is a single CSV file which contains 22 columns and 2939 rows. The columns include the country and year in which the data are taken, the life expectancy in years, some information about the country's health expenditure and GDP, and some health indicators (e.g., prevalence of 'thinness' in certain populations).

Each country contains yearly measurements between year 2000 and 2015. 

### Stationary Time Series

Because each country has measurements made in time (once per year), these data are time series. However, we made sure the data are approximately stationary -- the average change over time is small. **FILL IN SOME DETAILS HERE**

Therefore, the measurements for each country can be assumed independent. The interpretations from standard classifiers will not contain significant bias from the time-series correlations. We will continue with the analysis as if the year did not exist. 


## <a name="cleaning"></a>Cleaning

We simply drop the NaN's. In total, the Population column has the most NaN's -- 652 of them. That is 22% of the dataset. After removing _all_ of the NaN's, we are left with 1649 rows. We lost nearly half of the data because we drop any row containing at least one NaN. We could have filled the values in with the mean or median value, but it may not have improved the classifiers much. 

Our target column is the "Life expectancy " column (note the space). We transformed it into 0's and 1's -- values below the column's median are zero and values above are one.

The other columns we drop:

* The "Country" column -- because we did not want the classifier mapping a country name to the life expectancy. 
* The "Income composition of resources" column -- this is the Human Development Index (HDI) combines education, life expectency, and GNI index (economic factor). The column description was not entirely clear whether this was only the "income" portion of the HDI, or the full index, which is directly related to our target. We drop the column to circumvent any issue.
* The "Adult mortality" rate -- this is a generic column. We want actionable factors which contribute. Also, if more adults are dying (as opposed to children), the average life expectancy will be lower. It is directly correlated with our target column.
* The "Population" column **WE MAY PUT BACK**
* The "GDP" **WE MAY PUT BACK**



### Country Dependence

We have all of these numbers. Are they correlated with the country -- would a classifier simply be picking up the country, even after we removed the column. This issue is related to multicollinearity / bias. Implicitly, the country should have no effect. There are underlying causes (maybe disease or conflict). So if a combination of our remaining columns is correlated with the country, our model will be biased. 

To check this, we did two things. First, **WHAT HAPPENED**

Then, we split all of our data into two groups, each group with different countries. No country was in both datasets. We trained our classifiers on the first group, and predicted the life expectancy in the second group. The split was important -- the models had not been trained on the countries in the second group. The accuracy of the classifiers dropped by just 5%. There was an effect, but it was not very large, and might be simply because we're training the data on half the data. _We can split the data randomly in half and perform the same experiment to more accurately compare the accuracies_


## <a name="classify"></a>Classifier



## <a name="glossary"></a>Definitions

* "Low" life expectancy - life expectancy (in years) below the world-wide median life expectancy
* "High" life expectancy - life expectancy (in years) above the world-wide median life expectancy

## <a name="columns"></a>The Columns

Column names and descriptions come from Kaggle.

1. Country
2. Year
3. Status - "developed" or "developing" country
4. Life expectancy - in years
5. Adult Mortality - Adult Mortality Rates of both sexes (probability of dying between 15 and 60 years per 1000 population)
6. Infant deaths - Number of Infant Deaths per 1000 population
7. Alcohol - Alcohol, recorded per capita (15+) consumption (in litres of pure alcohol)
8. Percentage expenditure - Expenditure on health as a percentage of Gross Domestic Product per capita(%)
9. Hepatitis B - Hepatitis B (HepB) immunization coverage among 1-year-olds (%)
10. Measles - Measles - number of reported cases per 1000 population
11. BMI - Average Body Mass Index of entire population
12. Under-five deaths - Number of under-five deaths per 1000 population
13. Polio - Polio (Pol3) immunization coverage among 1-year-olds (%)
14. Total expenditure - General government expenditure on health as a percentage of total government expenditure (%)
15. Diphtheria - Diphtheria tetanus toxoid and pertussis (DTP3) immunization coverage among 1-year-olds (%)
16. HIV/AIDS - Deaths per 1 000 live births HIV/AIDS (0-4 years)
17. GDP - Gross Domestic Product per capita (in USD)
18. Population - Population of the country
19. Thinness  1-19 years - Prevalence of thinness among children and adolescents for Age 10 to 19 (% )
20. Thinness 5-9 years - Prevalence of thinness among children for Age 5 to 9(%)
21. Income composition of resources - Human Development Index in terms of income composition of resources (index ranging from 0 to 1)
22. Schooling - Number of years of Schooling(years)