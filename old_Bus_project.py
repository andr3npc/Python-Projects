# Import necessary libraries
import pandas as pd

# Load the data
businesses = pd.read_csv("data/businesses.csv")
new_businesses = pd.read_csv("data/new_businesses.csv")
countries = pd.read_csv("data/countries.csv")
categories = pd.read_csv("data/categories.csv")

# What is the oldest business on every continent?

# Start by merging the businesses and countries datasets into one
businesses_countries = businesses.merge(countries, on="country_code")

# Create a new DataFrame that lists only the continent and oldest year_founded
continent = businesses_countries.groupby("continent").agg({"year_founded":"min"})

# Merge this continent DataFrame with businesses_countries
merged_continent = continent.merge(businesses_countries, on=["continent", "year_founded"])

# Subset the continent DataFrame so that only the four columns of interest are included, saving it as oldest_business_continent
oldest_business_continent = merged_continent[["continent", "country", "business", "year_founded"]]

# View the result
print(oldest_business_continent)

# How many countries per continent lack data on the oldest businesses? 
# Does including the `new_businesses` data change this?

# Add the data in new_businesses to the existing businesses
all_businesses = pd.concat([new_businesses, businesses])

# Perform a new merge between the businesses and the countries data. Use additional parameters this time to perform an outer merge and create an indicator column to better see the missing values. An outer merge combines two DataFrames based on a key column and includes all rows from both DataFrames
new_all_countries = all_businesses.merge(countries, on="country_code", how="outer",  indicator=True)

# Filter to find countries with missing business data
new_missing_countries = new_all_countries[new_all_countries["_merge"] != "both"]

# Group by continent and create a "count_missing" column
count_missing = new_missing_countries.groupby("continent").agg({"country":"count"})
count_missing.columns = ["count_missing"]

# View the results
print(count_missing)

# Which business categories are best suited to last over the course of centuries?

# Start by merging the businesses and categories data into one DataFrame
businesses_categories = businesses.merge(categories, on="category_code")

# Merge all businesses, countries, and categories together
businesses_categories_countries = businesses_categories.merge(countries, on="country_code")

# Create the oldest by continent and category DataFrame
oldest_by_continent_category = businesses_categories_countries.groupby(["continent", "category"]).agg({"year_founded":"min"})
oldest_by_continent_category.head()
