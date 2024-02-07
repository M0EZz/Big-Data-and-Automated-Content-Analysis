#!/usr/bin/env python
# coding: utf-8

# # Take Home Exam - Big Data And Automated Content Analysis
# ## Helge Moes, 11348801

# # 0. Preliminary steps

# In[731]:


# The libraries are imported
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[732]:


# The json file is loaded into python
with open('takehome.json') as f:
    data = json.load(f)


# In[733]:


# Each record is converted in the list
# The 'json_normalize' pandas function converts JSON data into a pandas DataFrame 
from pandas import json_normalize

df = json_normalize(data)


# # 1.1 Explore overview of the dataset structure

# In this section the overview of the data structure is examined. It contains the first 5 rows of the table and gives a summary of the statistics and numeric columns and the missing values in each column.

# In[734]:


# The first 5 rows of the table are displayed
print(df.head())

# The '.describe' method gives descriptive information, such as count, mean, standard deviation, minimum and maximum values of the data frame
print(df.describe())

# The 'isnull()' method returns the presence of null values with True and a non-null value as False
print(df.isnull().sum())


# # 1.2 Cleaning the original table of Tripadvisor for exploratory analysis

# Based on the summary of the data, we can conclude that there are 31 hotels that are located in Amsterdam (since the file starts counting from 0). There are multiple steps that can be made in order to clean the table for the exploratory analysis. Moreover, the interesting variables for conducting an exploratory analysis are: name, reviewquantity and overall_rating. The steps can be summarized as followed:
# 
# Firstly, all the "META." can be left out of the table.
# 
# Secondly, the 'reviewquantity' can be converted into a column that contains a int data type, so it can be used for quantifiable examination.
# 
# It seems that 'reviews' is a nested list. In this data project, an exploratory analysis is run and the 'reviews' shall be written into a csv.file, so it can be read properly. This shall be conducted in chapter 3.

# In[735]:


# The columns that contain "META." hold no valuable information and can be filtered out
# Consequently, regex is used as an argument to filter a regular expression as a string
# '^(?!.*META).*$' may look weird, but the '^' gives the beginning of the string and the '$' indicates the end
# (?!.*META) gives a negative assertion that matches to any string that does not conatin 'META', where the '.*' matches any character 0 or more times in the entire string
df = df.filter(regex='^(?!.*META).*$')

# The 'reviewquantity' is converted into a column with a string type, this enables for string methods to be used
# The pandas method '.astype' converts the values in 'reviewquantity' into strings
df['reviewquantity'] = df['reviewquantity'].astype(str)

# The commas and ' reviews' text are removed from the 'reviewquantity' column
# The 'str.replace()' method is implemented to find and replace all occurences that ',' with an empty string
# In 'reviewquantity' ' reviews' is replaced with '' 
df['reviewquantity'] = df['reviewquantity'].str.replace(',', '').str.replace(' reviews', '')

# The 'reviewquantity' column is converted to an 'int' data type, so the number of reviews are quantifiable
df['reviewquantity'] = df['reviewquantity'].astype(int)

# Overview of the first 5 inputs of the cleaned dataset in table
df.head()


# It is evident that the column 'reviews' contains a list of reviews for each hotel that is nested in the dictionary.
# Each review is represented as a key that holds information as the date the review was posted, the username of the reviewer, the title of the review, etc.

# # 2.1 Exploratory Analysis Of The Original Data Frame - The Number Of Each Rating Value Being Counted

# In this exploratory analysis the frequency of ratings that are presented in the data (3.5, 4.0, 4.5) are counted and portrayed in a graph. This shows which rating is most presented in the data.

# In[736]:


# The number of each rating value is counted
# 'value_counts()' counts all the values that are represented in 'overall_rating'
rating_counts = df['overall_rating'].value_counts()

# The size is established and the style is set to 'darkgrid' as mentioned in https://seaborn.pydata.org/tutorial/aesthetics.html
plt.figure(figsize=(9, 7))
sns.set_style('darkgrid')

# A barplot is created of the number of ratings that is counted
# The color 'colorblind' is retrieved from: https://seaborn.pydata.org/tutorial/color_palettes.html
sns.barplot(x=rating_counts.index, y=rating_counts.values, palette="colorblind")

# The titles and labels of the axis are set
plt.title('Hotel Rating Counts', fontsize=20)
plt.xlabel('Rating', fontsize=15)
plt.ylabel('Count', fontsize=15)

# The plot is displayed.
plt.show()


# Explanation of the results:
# 
# The rating that is mostly counted is 4.0 (16 times). 
# The rating that is counted the least amount of times is 4.5 (6 times).
# This shows that it is challenging to get a rating of 4.5 on average in correlation with the other ratings, which are represented more in the data.

# # 2.2 Exploratory Analysis Of The Original Data Frame - Average Hotel Ratings

# In this exploratory analysis the average hotel ratings are displayed, in order to distinguish which hotel is averagely higher rated than the other.
# 

# In[737]:


# The average rating for each hotel is calculated and the values are sorted from highest to lowest average rated hotel by using the '.mean'
# The function 'df.groupby()' groups the 'name' by a value of the 'overall_rating'
# The function  '.sort_values()' sorts the values in an ascending order and is stored in the variable 'avg_ratings'
avg_ratings = df.groupby('name')['overall_rating'].mean().sort_values()

# The size and the style of the figure are established
# The style 'whitegrid' is chosen from https://seaborn.pydata.org/tutorial/aesthetics.html
plt.figure(figsize=(10, 6))
sns.set_style('whitegrid')

# A horizontal bar plot is created by using 'barh' of the average ratings, this choice is made to portray all the hotels and to clearly distinguish which hotel is higher rated than the other 
# This is retrieved from https://www.python-graph-gallery.com/ and adjusted to fit the code
plt.barh(avg_ratings.index, avg_ratings.values, color='Blue')

# The labels are set for the ratings and hotels
plt.title('Hotel Ratings (Average)', fontsize=20)
plt.xlabel('Rating', fontsize=15)
plt.ylabel('Name Hotel', fontsize=15)

# The grid lines are added
plt.grid(axis='x', alpha=1)

# Finally, the plot is displayed
plt.show()


# Explanation of the results:
# 
# Based on this visualization we can determine that 377 House - Amsterdam, WestCord Fashion Hotel Amsterdam, WestCord Art Hotel Amsterdam, Volkshotel, Crowne Plaza Amsterdam South and Hotel The neighbour's Magniolia are on average the highest rated with a rating of 4.5.
# Whereas; Mercure Hotel Amsterdam West, Hotel Atlas Vondelpark, Rembrandt Square Hotel Amsterdam, Hotel Abba, Rokin Hotel, The Alfred Hotel, Hampshire Hotel, Best Western Delphi Hotel and XO Hotels Blue Tower are rated the lowest on average with 3.5.

# # 2.3 Exploratory Analysis - Number Of Reviews Per Hotel

# In this exploratory analysis the number of reviews per hotel is examined in order to determine which hotel has received the most reviews in comparison to the others.

# In[738]:


# The 'reviewquantity' column is converted to an int data type, so the number of reviews are quantifiable
df['reviewquantity'] = df['reviewquantity'].astype(int)

# The data is sorted by 'reviewquantity', with the highest reviewed hotel mentioned first and the less reviewed as last
# The 'inplace=True' parameter allows sorts the data so it ascends
df.sort_values(by='reviewquantity', inplace=True)

# A horizontal bar plot is created by using 'barh' of the number of reviews per hotel, this choice is made to portray all the hotels and to clearly distinguish which hotel has the most reviews in comparison to the others
# The idea of the plot is retrieved from https://www.python-graph-gallery.com/ and adjusted to fit the code
plt.figure(figsize=(10, 8))
plt.barh(df['name'], df['reviewquantity'], color='Red')
plt.title('Number of Reviews per Hotel', fontsize=20)
plt.xlabel('Number of Reviews', fontsize=15)
plt.ylabel('Name Hotel', fontsize=15)

# The plot is printed
plt.show()


# Explanation of the results:
# 
# In this more detailed graph, it is apparent that WestCord Fashion Hotel Amsterdam received the most reviews. This might indicate that it had the most number of customers.
# The least reviewed hotel is Hotel Atlas Vondelpark, this might indicate that the hotel has not been visited often or the customers tend to not fill in a review on Tripadvisor.

# # 3.1 Extracting the keys from the 'reviews' column

# Before the 'reviews' column can be cleaned so it can be read properly, the keys need to be extracted from the data, so they can be utilized for a new list.

# In[739]:


# The keys of the 'reviews' list are extracted
# The 'set()' function creates a set object that ignores duplicate keys
# The '|=' is used to update the 'reviews_keys' with the keys in each dictionary
reviews_keys = set()
for item in data:
    for review in item.get('reviews', []):
        reviews_keys |= set(review.keys())

# The review keys are printed
print(reviews_keys)


# # 3.2 Cleaning 'reviews' column and writing it into a CSV file

# Now that the keys are extracted, a new list can be made in order to display the information that is in 'reviews'. Finally, the reviews are rewritten into a new CSV file: 'reviews_data.csv'.

# In[740]:


# A new list is created in order to store the data of the reviews column
reviews_data_list = []

# The items in the JSON file are looped
for item in data:
    # The reviews are looped into the item
    # The 'get()' method avoids raising an error if a certain key is not present in the dictionary and returns a default value
    for review in item.get('reviews', []):
        # The review data is extracted and added to the list that was presented in 'reviews_keys' and creates a new dictionary object called 'review_data'
        # Note: images is left out, since it did not hold any valuable information
        review_data = {
            'rating': review.get('rating'),
            'response_date': review.get('response_date'),
            'contributions': review.get('contributions'),
            'partnership': review.get('partnership'),
            'review': review.get('review'),
            'votes': review.get('votes'),
            'date': review.get('date'),
            'headline': review.get('headline'),
            'responder': review.get('responder'),
            'date_of_stay': review.get('date_of_stay'),
            'username': review.get('username'),
            'location': review.get('location'),
            'specific_ratings': review.get('specific_ratings'),
            'response': review.get('response'),
            'travel_company': review.get('travel_company'),
            'mobile': review.get('mobile'),
        }
        reviews_data_list.append(review_data)

# A pandas dataframe is created with the 'reviews' data
reviews_df = pd.DataFrame(reviews_data_list)

# The review data is written into a CSV file, where the 'index=False' does not include the index column to the CSV file
reviews_df.to_csv('reviews_data.csv', index=False)

reviews_df.head()


# # References

# Arcila, W. van A., Damian Trilling &. Carlos. (2022, March 11). Computational Analysis of Communication. https://cssbook.net/
# 
# Choosing color palettes—Seaborn 0.12.2 documentation. (n.d.). Retrieved May 10, 2023, from https://seaborn.pydata.org/tutorial/color_palettes.html
# 
# Holtz, Y. (n.d.). Control the color of barplots built with matplotlib. The Python Graph Gallery. Retrieved May 10, 2023, from https://www.python-graph-gallery.com/3-control-color-of-barplots/
