
# coding: utf-8

# # Online retail
# 
# ### a) Metadata
# 
# Link to the dataset: http://archive.ics.uci.edu/ml/datasets/Online+Retail
# 
# 
# This is a transnational data set which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail.The company mainly sells unique all-occasion gifts. Many customers of the company are wholesalers.
# 
# Attribute Information:
# 
# - InvoiceNo: Invoice number. Nominal, a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'c', it indicates a cancellation. 
# - StockCode: Product (item) code. Nominal, a 5-digit integral number uniquely assigned to each distinct product. 
# - Description: Product (item) name. Nominal. 
# - Quantity: The quantities of each product (item) per transaction. Numeric.	
# - InvoiceDate: Invice Date and time. Numeric, the day and time when each transaction was generated. 
# - UnitPrice: Unit price. Numeric, Product price per unit in sterling. 
# - CustomerID: Customer number. Nominal, a 5-digit integral number uniquely assigned to each customer. 
# - Country: Country name. Nominal, the name of the country where each customer resides.
# 
# 
# Relevant Papers:
# 
# - The evolution of direct, data and digital marketing, Richard Webber, Journal of Direct, Data and Digital Marketing Practice (2013) 14, 291â€“309. 
# - Clustering Experiments on Big Transaction Data for Market Segmentation, 
# - Ashishkumar Singh, Grace Rumantir, Annie South, Blair Bethwaite, Proceedings of the 2014 International Conference on Big Data Science and Computing. 
# - A decision-making framework for precision marketing, Zhen You, Yain-Whar Si, Defu Zhang, XiangXiang Zeng, Stephen C.H. Leung c, Tao Li, Expert Systems with Applications, 42 (2015) 3357â€“3367.
# 
# 
# ### b) Goal of RFM analysis
# Recency-Frequency-Monetary analysis is very well used in econometrics to understand customer behavior. 

# ## 1) Data discovery & cleaning

# In[1]:


import pandas as pd 
import numpy as np
import seaborn as sns 

import matplotlib.pyplot as plt
FIGSIZE = (12.0, 5.0)
plt.rcParams['figure.figsize'] = (12.0, 5.0)

pd.options.display.max_rows = 55
pd.options.display.max_colwidth = 500


# In[2]:


link = 'Online Retail.xlsx'


# In[3]:


xl = pd.ExcelFile(link)

for sheet in xl.sheet_names[1:]:
    print(xl.parse(sheet).columns[0])


# In[4]:


on = pd.ExcelFile(link)

# Print the sheet names
#print(on.sheet_names)


# In[57]:


# Load a sheet into a DataFrame by name: df1
onli = on.parse('Online Retail')


# In[56]:


onli.head(2)


# In[7]:


onli.shape


# In[8]:


#Checking null values
onli.isnull().sum(axis=0)


# There are 135080 CustomerID´s missing out of 541909. The analysis is based on customers, then let´s remove these rows from the dataframe. 

# In[9]:


onli = onli[pd.notnull(onli['CustomerID'])]


# In[10]:


onli.describe()


# In[11]:


# Check the minimum values in UnitPrice and Quantity columns.

onli.Quantity.min() 
# THe result is a negative value, which is impossible. 

#Only positive quantities
onli = onli[(onli['Quantity']>0)]
onli.shape 


# #### Grouping by country 
# 
# Customer clusters can vary by geography. Most of customers come from the UK with 3950 customers, the second biggest market is Germany with only 95 customers. It makes sense to select only customers from the UK, but we want to find the top ten customers, so we leave all customers.

# In[12]:


#Grouping by country. 
customer_country= onli[['Country','CustomerID']].drop_duplicates()

customer_country.groupby(['Country'])['CustomerID'].aggregate('count').reset_index().sort_values('CustomerID', ascending=False)


# In[13]:


#online= onli.loc[onli['Country'] == 'United Kingdom']


# In[14]:


online.shape


# #### Check unique value for all columns

# In[15]:


def unique_counts(columns):
   for i in online.columns:
       count = online[i].nunique()
       print(i, ": ", count)
unique_counts(online)


# ## 2) Cohort Analysis and RFM score
# 
# Let´s compare metrics across customer lifecycle, distinguising them by cohort, where 'CohortMonth' is the time the customer bought something for the first time. We start the  analysis setting 10/12/2011 as the first day of acquisition (defined below as the NOW variable). 

# In[16]:


import datetime as dt
from datetime import datetime


# In[17]:


#Total spend
online['TotalPrice'] = online['Quantity'] * online['UnitPrice']

#First purchase
online['InvoiceDate'].min()

#Last order
online['InvoiceDate'].max()


# In[18]:


import datetime as dt
NOW = dt.datetime(2011,12,10)
online['InvoiceDate'] = pd.to_datetime(online['InvoiceDate'])


# In[19]:


rfmTable = online.groupby('CustomerID').agg({'InvoiceDate': lambda x: (NOW - x.max()).days, 
                                             'InvoiceNo': lambda x: len(x), 
                                             'TotalPrice': lambda x: x.sum()})

rfmTable['InvoiceDate'] = rfmTable['InvoiceDate'].astype(int)
rfmTable.rename(columns={'InvoiceDate': 'recency', 
                         'InvoiceNo': 'frequency', 
                         'TotalPrice': 'monetary'}, inplace=True)


# In[20]:


rfmTable.head()


# Finding particular customers

# In[21]:


first_customer = online[online["CustomerID"]== 12346.0]

first_customer


# ### 2.b) Split the matrix 
# 
# To understand customers, we give a score to R, F and M. The procedure is to use a range of values, or quantiles, that goes from lower to higher values. In this way we will have a segmented RFM table that ranges between 111 (higher) and 555 (lower).

# In[22]:


quantiles = rfmTable.quantile(q=[0.80,0.60,0.40, 0.20])
quantiles = quantiles.to_dict()


# In[23]:


segmented_rfm = rfmTable


# In[24]:


def RScore(x,p,d):
    if x <= d[p][0.80]:
        return 1
    elif x <= d[p][0.60]:
        return 2
    elif x <= d[p][0.40]: 
        return 3
    elif x <= d[p][0.20]: 
        return 4
    else:
        return 5
    
def FMScore(x,p,d):
    if x <= d[p][0.80]:
        return 5
    elif x <= d[p][0.60]:
        return 4
    elif x <= d[p][0.40]: 
        return 3
    elif x <= d[p][0.20]: 
        return 4
    else:
        return 1


# In[27]:


# Add segment numbers to the newly created segmented RFM table

segmented_rfm['R'] = segmented_rfm['recency'].apply(RScore, args=('recency',quantiles,))
segmented_rfm['F'] = segmented_rfm['frequency'].apply(FMScore, args=('frequency',quantiles,))
segmented_rfm['M'] = segmented_rfm['monetary'].apply(FMScore, args=('monetary',quantiles,))
segmented_rfm.head()


# In[77]:


segmented_rfm['RFMScore'] = segmented_rfm.R.map(str) + segmented_rfm.F.map(str) + segmented_rfm.M.map(str)

segmented_rfm.head()


# In[78]:


segmented_rfm.groupby('RFMScore')['RFMScore'].count()


# ### 2.c) Who are the top 10 of our best customers!

# In[79]:


segmented_rfm[segmented_rfm['RFMScore']=='555'].sort_values('monetary', ascending=False).head(10)


# ## 3) Another approach for cohort analysis and quantile score
# 
# This is a more Pythonistic approach.

# In[32]:


# Define a function that will parse the date
def get_month(x): return dt.datetime(x.year, x.month, 1) 

# Create InvoiceMonth column
online['InvoiceMonth'] = online['InvoiceDate'].apply(get_month) 

# Group by CustomerID and select the InvoiceMonth value
grouping = online.groupby('CustomerID')['InvoiceMonth'] 

# Assign a minimum InvoiceMonth value to the dataset
online['CohortMonth'] = grouping.transform('min')

# View the top 5 rows
online.shape

online.head(5)


# In[33]:


def get_date_int(df, column):
    year = df[column].dt.year
    month = df[column].dt.month
    #day = df[column].dt.day
    return year, month


# In[34]:


# Get the integers for date parts from the `InvoiceMonth` column
invoice_year, invoice_month = get_date_int(online, 'InvoiceMonth')

# Get the integers for date parts from the `CohortMonth` column
cohort_year, cohort_month = get_date_int(online, 'CohortMonth')


# Apply time offset value calculating the difference between the Invoice and Cohort dates in years, months and days.

# In[35]:


# Calculate difference in years
years_diff = invoice_year - cohort_year

# Calculate difference in months
months_diff = invoice_month - cohort_month

# Calculate difference in days
#days_diff = invoice_day - cohort_day

# Extract the difference in days from all previous values
#online['CohortIndexdays'] = years_diff * 365 + months_diff * 30 + days_diff + 1

# Extract the difference in months  from all previous values
online['CohortDifference'] = years_diff * 365 + months_diff * 30 + 1

online.head(2)


# ### 3.a) Customer retention metrics
# 
# Let's count monthly active customers from each cohort. The amount of costumers during the first month is the size of the cohort. 

# In[36]:


grouping = online.groupby(['CohortMonth','CohortDifference'])

# Count the number of unique values per customer ID
cohort_data = grouping['CustomerID'].apply(pd.Series.nunique).reset_index().sort_values(by=['CustomerID'])

cohort_data.head()


# In[37]:


# Create a pivot to see the active customers per cohort. Recent cohorts will have more NaN.
cohort_counts = cohort_data.pivot(index='CohortMonth', columns='CohortDifference', values='CustomerID')

cohort_counts


# In[38]:


#The amount of costumers during the first month is the reference size of the cohort. 

cohort_sizes = cohort_counts.iloc[:,0]

cohort_sizes


# In[39]:


plt.plot(cohort_sizes)

plt.title('Cohort size')
plt.xlabel('Cohort Month')
plt.ylabel('Size')

plt.show()


# In[40]:


#What does axis=0 mean again? 

#Retention rate in %
retention = cohort_counts.divide(cohort_sizes, axis=0).round(3)*100

retention


# In[41]:


plt.figure(figsize=(10, 8))
plt.title('Retention rates')

sns.heatmap(data = retention, 
            annot = True, 
            fmt = '.0%', 
            vmin = 0.0, 
            vmax = 1, 
            cmap = 'BuGn')

plt.show()


# ### 3.b) Average quantity per cohort

# In[42]:


online.head(2)


# In[43]:


grouping = online.groupby(['CohortMonth', 'CohortDifference'])

cohort_data = grouping['Quantity'].mean()

cohort_data.head()


# In[44]:


cohort_data = cohort_data.reset_index()

cohort_data


# In[45]:


average_quantity = cohort_data.pivot(index='CohortMonth', 
                                     columns='CohortDifference', 
                                     values='Quantity')

average_quantity = average_quantity.round(1)

average_quantity.head()


# In[46]:


average_quantity.info()


# In[47]:


# Initialize an 8 by 6 inches plot figure
plt.figure(figsize=(8, 6))

plt.title('Average quantity bought by Monthly Cohorts')

sns.heatmap(data=average_quantity, annot=True, cmap='Blues')
plt.show()


# ### 3.c) Average unit price per cohort

# In[48]:


grouping = online.groupby(['CohortMonth', 'CohortDifference'])

cohort_dataunit= grouping['UnitPrice'].count().reset_index()

average_unitprice = cohort_dataunit.pivot(index='CohortMonth', 
                                     columns='CohortDifference', 
                                     values='UnitPrice')

average_unitprice = average_unitprice.round(1)

average_unitprice.head() 


# In[49]:


# Initialize an 8 by 6 inches plot figure
plt.figure(figsize=(16, 12))

plt.title('Average Spend by Monthly Cohorts')

sns.heatmap(data=average_unitprice, annot=True, cmap='Blues')
plt.show()


# ## 3) RFM segmentation based on quartile values

# ###### 1) Monetary (series)

# In[50]:


#Monetary 

online['spend'] = online['UnitPrice']*online['Quantity']  
money = online.groupby('CustomerID')['spend'].sum()

type(money)


# ###### 2) Recency (series)
# 
# Since it has to be calculated from a point in time, we can use as a reference the last invoice date, which is 2011-12-09 plus one day, 2011-12-10. The first invoice in the data was on 2010-12-01.

# In[51]:


# Define a function that will parse the date
def get_day(x): return dt.datetime(x.year, x.month, x.day) 

# Create InvoiceMonth column
online['InvoiceDay'] = online['InvoiceDate'].apply(get_day) 


# In[52]:


online['InvoiceDay'].min()
online['InvoiceDay'].max()

timeref =  dt.datetime(2011,12,10)

lastpurchase = online.groupby('CustomerID')['InvoiceDate'].max() 

recency = timeref - lastpurchase

type(recency)
recen = recency.to_frame(name=None).reset_index()

recen.rename(columns={'InvoiceDate': 'recency'}, inplace=True)


# ### Using quartiles to create recency labels
# 
# These labels can let us recognize the retention of customers, if they are active or casual buyers. 
# 
# The procedure is to sort customers base on a given metric, break customers into a pre-defined number of groups of equal size and assign a label to each group. 

# In[53]:


# Create string labels
r_labels = ['Active','Lapsed','Inactive','Churned']
# Divide into groups based on quartiles
recency_quartiles = pd.qcut(recen['recency'], q=4, labels=r_labels)
# Create new column
recen['Recency_Quartile'] = recency_quartiles
# Sort values from lowest to highest
recen.sort_values('recency')[:5]


# In[54]:


recen.groupby('Recency_Quartile').count()


# ###### 3) Frequency (series)

# In[55]:


freq = online.groupby('CustomerID')['InvoiceNo'].count()


# ## RFM Table

# In[58]:


rfmTable = online.groupby('CustomerID').agg({'InvoiceDate': lambda x: (timeref - x.max()).days,
                                             'InvoiceNo': lambda x: len(x), #'InvoiceNo': 'count'
                                             'spend': lambda x: x.sum()}) #'spend': 'sum'

rfmTable['InvoiceDate'] = rfmTable['InvoiceDate'].astype(int)

rfmTable.rename(columns={'InvoiceDate': 'recency',
                         'InvoiceNo': 'frequency',
                         'spend': 'monetary'}, inplace=True)

rfmTable.head(2)


# In[59]:


rfmTable = segmented_rfm


# ### Using quartiles to create  recency, frequency and monetary labels
# 
# The best customers have the lowest recency, higher frequency and higher monetary values. 

# In[60]:


# Add segment numbers to the newly created segmented RFM table

segmented_rfm['R'] = segmented_rfm['recency'].apply(RScore, args=('recency',quantiles,))
segmented_rfm['F'] = segmented_rfm['frequency'].apply(FMScore, args=('frequency',quantiles,))
segmented_rfm['M'] = segmented_rfm['monetary'].apply(FMScore, args=('monetary',quantiles,))

segmented_rfm.head()


# In[64]:


segmented_rfm2 = rfmTable

# Create labels for Recency (from low to high) and Frequency (this can be used to monetary as well)
r_labels = range(5, 0, -1); f_labels = range(1, 6)

# Assign these labels to three equal percentile groups 
r_groups = pd.qcut(segmented_rfm2['recency'], q=5, labels=r_labels)
f_groups = pd.qcut(segmented_rfm2['frequency'], q=5, labels=f_labels)
m_groups = pd.qcut(segmented_rfm2['monetary'], q=5, labels=f_labels)

# Create new columns R and F 
segmented_rfm2 = segmented_rfm2.assign(R=r_groups.values, F=f_groups.values, M= m_groups.values)


# In[65]:


def join_rfm(x): 
    return str(x['R']) + str(x['F']) + str(x['M'])

segmented_rfm2['RFM_Segment'] = segmented_rfm2.apply(join_rfm, axis=1)
segmented_rfm2['RFM_Score'] = segmented_rfm2[['R','F','M']].sum(axis=1)


segmented_rfm.head(2)


# ## 4) Metrics per RFM Score
# 

# #### Size of the segments 

# In[66]:


segments = segmented_rfm2.groupby('RFM_Segment').size().sort_values(ascending=False)[:5]

segments


# - Using 4 segments: The bigger segment has the lowes recency score (444) and the second biggest segment the highest recency score (111).
# - Using 5 segments: Bigger segment 555 score, second biggest 111.

# #### Top 10 customers 

# In[67]:


segmented_rfm2[segmented_rfm2['RFM_Segment'] =='555'].sort_values('monetary', ascending=False).head(3)


# #### Mean values

# In[68]:


RFM = segmented_rfm2

RFM.head()


# There are twelve RFM_Score groups, from 3 to 15.

# In[69]:


RFM.groupby('RFM_Score').agg({'recency': 'mean','frequency': 'mean','monetary': ['mean','count'] }).round(1)


# ### RFM levels: From platinum to pink!  
# 
# 'RFM_Score' goes from 3 to 15. 

# In[70]:


# Define rfm_level function
def rfm_level(df):
    if df['RFM_Score'] >= 14:
        return 'Platinum'
    elif ((df['RFM_Score'] >= 11) and (df['RFM_Score'] < 14)):
        return 'Gold'
    elif ((df['RFM_Score'] >= 8) and (df['RFM_Score'] < 11)):
        return 'Silver'
    elif ((df['RFM_Score'] >= 5) and (df['RFM_Score'] < 8)):
          return 'Bronce'
    else:
        return 'Pink'


# In[71]:


# Create a new variable RFM_Level
RFM['RFM_level'] = RFM.apply(rfm_level, axis=1)

RFM.head(5)


# In[73]:


#The last column "sum" gives the size of the level

RFM.groupby('RFM_level').agg({'recency': 'mean','frequency': 'mean','monetary': ['mean','sum'] }).round(1)


# Platinum customers are the top customers, followed by gold. 

# In[74]:


RFM.describe()


# ### RFM visualization 

# In[80]:


# Plot distributions 
plt.subplot(3, 1, 1); sns.distplot(RFM['recency'])
plt.subplot(3, 1, 2); sns.distplot(RFM['frequency'])
plt.subplot(3, 1, 3); sns.distplot(RFM['monetary'])

# Show the plot
plt.show()


# It is obvious from the result that frequency and monetary are very skewed. 
# 
# ## RFM pipeline 
# 
# The following pipeline needs to be applied:
# 1) To solve the skeweness:
# 2) Standarization: 
# 3) 
# 
# We apply logarithm to solve this.

# In[ ]:


#We need to change 'monet_log'. Apparently it has negative values
#Checking null values

#df.dropna(subset=['name', 'born'])
#RFM['monetary'].dropna(inplace=True)
RFM.isnull().sum()
RFM.isna().sum()


# In[ ]:


# Apply log transformation and unskewing.
RFM['freq_log'] = np.log(RFM['frequency'])
RFM['monet_log'] = np.log(RFM['monetary'])

RFM.dropna(subset=['monet_log', 'freq_log'], axis = 0 , inplace= True)

RFM.describe()


# In[ ]:


RFM['monet_log'] = RFM['monet_log'].astype(str).astype(float)

RFM.info()


# In[ ]:


plt.subplot(2, 1, 1) ; sns.distplot(RFM['freq_log'])
plt.subplot(2, 1, 2) ; sns.distplot(RFM['monet_log'])
plt.show()


# # 5) K means clustering 
# 
# ### Giving the same mean and variance to each variable
# Otherwise Kmeans do not work well.
# 
# Doing it by hand, obtaining RFM_norm dataset. 

# In[ ]:


justrfm = RFM[['recency', 'freq_log', 'monet_log']]

# Normalize the data by applying both centering and scaling
RFM_norm = (justrfm - justrfm.mean())/ justrfm.std()

RFM_norm.describe().round(2)


# *Same stuff but in one sequence* : 1) Log, 2) using a Scaler to normalize centering and scaling, 3) storing a new df: RFM2.

# In[ ]:


from sklearn.preprocessing import StandardScaler

# Initialize a scaler
scaler = StandardScaler()


# In[ ]:


just = RFM[['recency', 'frequency', 'monetary']]

just = just[just >= 1]
just.dropna(axis = 0 , inplace= True)

just.head()


# In[ ]:


#First | log 
rfm_log = np.log(just)

#Second | normalization with a scaler
#It doesn't work. ValueError: Input contains NaN, infinity or a value too large for dtype('float64').

scaler.fit(rfm_log)

#Third | saving in a new df
RFM2 = scaler.transform(rfm_log)
RFM2 = pd.DataFrame(RFM2, index= just.index, columns = just.columns)

RFM2.head(1)


# In[ ]:


plt.subplot(2, 1, 1) ; sns.distplot(rfm_log['frequency'])
plt.subplot(2, 1, 2) ; sns.distplot(rfm_log['monetary'])
plt.show()


# ### Methods to define the number of clusters 
# 
# Elbow criterion, mathematical methods (silhouette coefficients), experimentation and interpretation.

# In[ ]:


# Import KMeans 
from sklearn.cluster import KMeans


# #### Elbow criterion method

# In[ ]:


sse = {}

# Fit KMeans and calculate SSE for each k, we run a for loop with a random range 
for k in range(1, 21):
  
    # Initialize KMeans with k clusters
    kmeans = KMeans(n_clusters=k, random_state=1)
     
    # Fit KMeans on the normalized dataset
    kmeans.fit(RFM2)
    
    # Assign sum of squared distances to k element of dictionary
    sse[k] = kmeans.inertia_


# In[ ]:


# Add the plot title "The Elbow Method"
plt.title('The Elbow Method')

# Add X-axis label "k"
plt.xlabel('k')

# Add Y-axis label "SSE"
plt.ylabel('SSE')

# Plot SSE values for each key in the dictionary
sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
plt.show()


#  We can consider that 4 or 5 clusters is were there is the biggest angle.

# In[ ]:


kmeans = KMeans(n_clusters=5, random_state=1)  


# In[ ]:


#Compute k-means clustering on pre-processed data
kmeans.fit(RFM2) 

#Extract cluster labels from labels_ attribute
cluster_labels = kmeans.labels_ 


# In[ ]:


#Create a cluster label column in the original DataFrame:

RFMclustering = just.assign(Cluster = cluster_labels) 

RFMclustering.head()


# In[ ]:


#Calculate average RFM values and segment sizes per cluster value

sizecluster = RFMclustering.groupby(['Cluster']).agg({'recency': 'mean', 
                                        'frequency': 'mean',     
                                        'monetary': ['mean', 'count']}).round(0) 

sizecluster


# In[ ]:


#Using 4 clusters 
kmeans = KMeans(n_clusters=5, random_state=1)  
kmeans.fit(RFM2) 

cluster_labels = kmeans.labels_ 

RFMclustering = just.assign(Cluster = cluster_labels) 

sizecluster = RFMclustering.groupby(['Cluster']).agg({'recency': 'mean', 
                                        'frequency': 'mean',     
                                        'monetary': ['mean', 'count']}).round(0) 

sizecluster


# We decide the number of clusters according to the type of action we would like to perform, e.g. a marketing campaign. When we select 5 clusters, the smallest segment is around half the size of the biggest segment, a fair size to consider that 5 clusters is a good option.

# # 6) Segment interpretation
# 
# Approaches to build customer personas: 
# - Summary statistics for each cluster e.g. average RFM values (just done).
# - Snake plots (from market research)
# - Relative importance of cluster attributes compared to population
# 
# #### Prepare data for snake plots

# In[ ]:


# Transform datamart_normalized as DataFrame and add a Cluster column

snake = pd.DataFrame(RFM2, 
                     index=just.index,
                     columns=just.columns) 

snake['Cluster'] = RFMclustering['Cluster']

snake.head()


# In[ ]:


#Melt the data into a long format so RFM values and metric names are stored in 1 column each


# Melt the normalized dataset and reset the index
snake_melt = pd.melt(snake.reset_index(),
                        
# Assign CustomerID and Cluster as ID variables
                    id_vars=['CustomerID', 'Cluster'],

# Assign RFM values as value variables
                    value_vars=['recency', 'frequency', 'monetary'], 
                        
# Name the variable and value
                    var_name ='Metric', value_name ='Value')


# In[ ]:


snake_melt.head(5)


# In[ ]:


# Add the plot title
plt.title('Snake plot of normalized variables')

# Add the x axis label
plt.xlabel('Metric')

# Add the y axis label
plt.ylabel('Value')

# Plot a line for each value of the cluster variable
sns.lineplot(data= snake_melt, x='Metric', y='Value', hue='Cluster')

plt.show()


# In[ ]:


# Calculate average RFM values for each cluster
cluster_avg = RFMclustering.groupby(['Cluster']).mean() 

# Calculate average RFM values for the total customer population
population_avg = just.mean()

# Calculate relative importance of cluster's attribute value compared to population
relative_imp = cluster_avg / population_avg - 1

# Print relative importance scores rounded to 2 decimals
relative_imp.round(2)


# In[ ]:


# Initialize a plot with a figure size of 8 by 2 inches 
plt.figure(figsize=(10, 4))

# Add the plot title
plt.title('Relative importance of attributes')

# Plot the heatmap
sns.heatmap(data=relative_imp, annot=True, fmt='.2f', cmap='RdYlGn')
plt.show()

