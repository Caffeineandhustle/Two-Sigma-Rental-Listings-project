#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Problem Statement : you will predict how popular an apartment rental listing is based on the 
#listing content like text description, photos, number of bedrooms, price, etc. 
#The data comes from renthop.com, an apartment listing website. These apartments are located in New York City.

#The target variable, interest_level, is defined by the number of 
#inquiries a listing has in the duration that the listing was live on the site.


# # Data Exploration

# In[22]:


#import necessary libraries

#Pandas :  for data analysis
#numpy : for Scientific Computing.
#matplotlib and seaborn : for data visualization
#scikit-learn : ML library for classical ML algorithms
#math :for mathematical functions


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


get_ipython().run_line_magic('matplotlib', 'inline')

color=sns.color_palette()

pd.options.mode.chained_assignment = None


# In[17]:


#Read train Data 

train_df =pd.read_json("/Users/priyeshkucchu/Desktop/ts/train.json")


# In[6]:


# Print top rows

train_df.head()


# In[7]:


# Print last rows

train_df.tail()


# In[8]:


#Shows stats for your data

train_df.describe()


# In[9]:


#Show complete information about your data 

train_df.info()


# In[10]:


#Check for nulls

train_df.isnull().max()


# In[12]:


#Check no of rows and columns in the dataset

train_df.shape


# In[18]:


#Check value counts of column bedrooms
train_df.bedrooms.value_counts()


# In[19]:


#Check value counts of column Interest Level
train_df.interest_level.value_counts()


# In[18]:


#Read test data

test_df =pd.read_json("/Users/priyeshkucchu/Desktop/ts/test.json")


# In[21]:


#Show complete information of your test dataset
test_df.info()


# In[22]:


#Show top rows
test_df.head()


# In[23]:


#Show last rows
test_df.tail()


# In[24]:


#Check for nulls

test_df.isnull().max()


# In[25]:


#Check for no of rows and columns for test data set

test_df.shape


# In[27]:


#Check value counts of column bedrooms
test_df.bedrooms.value_counts()


# # Data Visualization

# In[29]:


# Plot interest level

inter_level= train_df['interest_level'].value_counts()

plt.figure(figsize=(8,4))
sns.barplot(inter_level.index,inter_level.values, alpha=0.8, color="blue")
plt.xlabel("Interest Level by category")
plt.ylabel("No of ocurrances")
plt.title("Interest level chart")


# Interest level is low for most of the cases followed by medium and then high which makes sense.

# # Explore numerical features

# Numerical features are
# 
# 1.bathrooms
# 2.bedrooms
# 3.price
# 4.latitude
# 5.longitude

# In[31]:


#Plot numerical features mentioned above

bathroom_counts=train_df["bathrooms"].value_counts()

plt.figure(figsize=(8,4))
sns.barplot(bathroom_counts.index,bathroom_counts.values,alpha=0.8,color="orange")
plt.xlabel("Bathroom Counts")
plt.ylabel("No of occurances")
plt.title("Bathroom count chart")


# In[39]:


# bathroom Counts vs Interest Level

train_df['bathrooms'].loc[train_df["bathrooms"]>3]=3

plt.figure(figsize=(12,8))
sns.violinplot(x='interest_level',y="bathrooms",data=train_df,palette="winter", inner="box")
plt.xlabel("Interest Level")
plt.ylabel("Bathrooms")
plt.title("Bathroom Counts vs Interest Level")
plt.show()


# In[40]:


#show stats of train data set
train_df.describe()


# In[41]:


train_df.bathrooms.value_counts()


# In[43]:


#Bedroom Counts

bedroom_counts=train_df["bedrooms"].value_counts()

plt.figure(figsize=(8,4))
sns.barplot(bedroom_counts.index,bedroom_counts.values,alpha=0.8,color="blue")
plt.xlabel("Bedroom Counts")
plt.ylabel("No of occurances")
plt.title("Bedroom count chart")


# In[44]:


# bedroom Counts vs Interest Level

train_df['bathrooms'].loc[train_df["bedrooms"]>3]=3

plt.figure(figsize=(12,8))
sns.violinplot(x='interest_level',y="bedrooms",data=train_df,palette="Set2", inner="box")
plt.xlabel("Interest Level")
plt.ylabel("Bedrooms")
plt.title("Bedroom Counts vs Interest Level")
plt.show()


# In[45]:


train_df.bedrooms.value_counts()


# In[46]:


plt.figure(figsize=(8,4))
sns.countplot(x='bedrooms', hue='interest_level',data=train_df)
plt.ylabel('Number of Occurrences', fontsize=10)
plt.xlabel('bedrooms', fontsize=10)
plt.title("Bedrooms counts by interest level chart ")
plt.show()


# # Price Variable Distribution

# In[49]:


plt.figure(figsize=(12,8))
plt.scatter(range(train_df.shape[0]),np.sort(train_df.price.values))
plt.xlabel("Index")
plt.ylabel("Price")
plt.title("Price Variable Distribution")
plt.show()


# Looks like there are some outliers in this feature. So let us remove them and then plot again.

# In[51]:


upper_limit=np.percentile(train_df.price.values,99)
train_df['price'].loc[train_df['price']>upper_limit]=upper_limit

plt.figure(figsize=(12,8))
sns.distplot(train_df.price.values,bins=50,kde=True)
plt.xlabel('price', fontsize=12)
plt.show()


# Observation : The distribution is right skewed

# # Latitude and Longitude

# In[19]:


lower_limit=np.percentile(train_df.latitude.values,1)
upper_limit=np.percentile(train_df.latitude.values,99)
train_df['latitude'].loc[train_df['latitude']>upper_limit]=upper_limit
train_df['latitude'].loc[train_df['latitude']<lower_limit]=lower_limit

plt.figure(figsize=(12,8))
sns.distplot(train_df.latitude.values,bins=50,kde=True)
plt.xlabel('latitude', fontsize=12)
plt.show()


# Observation : So the latitude values are primarily between 40.6 and 40.9

# In[20]:


lower_limit=np.percentile(train_df.longitude.values,1)
upper_limit=np.percentile(train_df.longitude.values,99)
train_df['longitude'].loc[train_df['longitude']>upper_limit]=upper_limit
train_df['longitude'].loc[train_df['longitude']<lower_limit]=lower_limit

plt.figure(figsize=(12,8))
sns.distplot(train_df.longitude.values,bins=50,kde=True)
plt.xlabel('longitude', fontsize=12)
plt.show()


# Observation : The longitude values range between -73.8 and -74.02. So the data corresponds to the New York City.

# # Map 

# In[25]:


from mpl_toolkits.basemap import Basemap
from matplotlib import cm


west, south, east, north = -74.02, 40.64, -73.85, 40.86

fig = plt.figure(figsize=(14,10))
ax=fig.add_subplot(111)
m = Basemap(projection='merc', llcrnrlat=south, urcrnrlat=north,
            llcrnrlon=west, urcrnrlon=east, lat_ts=south, resolution='i')
x, y = m(train_df['longitude'].values, train_df['latitude'].values)

m.hexbin(x, y, gridsize=200,bins='log', cmap=cm.YlOrRd_r);


# In[28]:


# Take a look at Date created

train_df['created']=pd.to_datetime(train_df['created'])
train_df['created_date']= train_df['created'].dt.date

count_srs= train_df['created_date'].value_counts()

plt.figure(figsize=(10,6))

ax= plt.subplot(111)
ax.bar(count_srs.index,count_srs.values,alpha=0.8)
ax.xaxis_date()
plt.xticks(rotation='vertical')
plt.show()


# In[29]:


test_df['created']=pd.to_datetime(test_df['created'])
test_df['created_date']= test_df['created'].dt.date

count_srs= test_df['created_date'].value_counts()

plt.figure(figsize=(10,6))

ax= plt.subplot(111)
ax.bar(count_srs.index,count_srs.values,alpha=0.8)
ax.xaxis_date()
plt.xticks(rotation='vertical')
plt.show()


# Observation: Both train and test has data from April to June 2016

# # Hour Created

# In[32]:


train_df['hour_created']=train_df['created'].dt.hour



count_srs= train_df['hour_created'].value_counts()

plt.figure(figsize=(10,6))

plt.figure(figsize=(12,6))
sns.barplot(count_srs.index, count_srs.values, alpha=0.8, color=color[3])
plt.xticks(rotation='vertical')
plt.xlabel("Hours")
plt.ylabel("No of listings")
plt.show()


# Observation : Listings are created during the early hours of the day (1 to 7am)

# # Photos

# In[34]:


train_df["n_pics"]=train_df['photos'].apply(len)
count_srs= train_df['n_pics'].value_counts()

plt.figure(figsize=(10,6))
sns.barplot(count_srs.index,count_srs.values,alpha=0.8)
plt.xlabel('Number of Photos', fontsize=10)
plt.ylabel('Number of Occurrences', fontsize=10)
plt.show()


# In[35]:


train_df['n_pics'].loc[train_df['n_pics']>12]=12
plt.figure(figsize=(10,6))
sns.violinplot(x='n_pics',y='interest_level', data=train_df,order=['low','medium','high'])
plt.xlabel("No of photos")
plt.ylabel("Interest Level")
plt.title("No of photos vs interest level")
plt.show()


# # Features

# In[36]:


train_df["n_features"]=train_df['features'].apply(len)
count_srs= train_df['n_features'].value_counts()

plt.figure(figsize=(10,6))
sns.barplot(count_srs.index,count_srs.values,alpha=0.8)
plt.xlabel('Number of Features', fontsize=10)
plt.ylabel('Number of Occurrences', fontsize=10)
plt.show()


# In[44]:


train_df['n_features'].loc[train_df['n_features']>17]=17
plt.figure(figsize=(12,10))
sns.violinplot(y='n_features',x='interest_level',orient="v", data=train_df,order=['low','medium','high'])
plt.xlabel("No of Features")
plt.ylabel("Interest Level")
plt.title("No of Features vs interest level")
plt.show()


# # Word Clouds

# In[49]:


from wordcloud import WordCloud

t_f=''
t_a=''
#t_desc=''

for ind,row in train_df.iterrows():
    for feature in row['features']:
        t_f = "".join([t_f,"_".join(feature.strip().split(" "))])
    t_a="".join([t_a,"_".join(row['display_address'].strip().split(" "))])
    #t_desc="".join([t_desc,row['description']])
t_f=t_f.strip()
t_a=t_a.strip()
#t_desc=t_desc.strip()

#Wordcloud for features

plt.figure(figsize=(14,6))
wordcloud=WordCloud(background_color='white',width=900,height=400,max_words=40,max_font_size=60).generate(t_f)
wordcloud.recolor(random_state=0)
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Wordcloud for Features")
plt.show()


#Wordcloud for Displayaddress

plt.figure(figsize=(14,6))
wordcloud=WordCloud(background_color='white',width=900,height=400,max_words=40,max_font_size=60).generate(t_a)
wordcloud.recolor(random_state=0)
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Wordcloud for Features")
plt.show()




# In[ ]:




