import pandas as pd


# load the dataset
df = pd.read_csv("mall_customers.csv")


# quick look at the data
print("Shape of dataset:", df.shape)
print("\nColumn names:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nBasic statistics (numeric columns):")
print(df.describe())

print("\nGender distribution:")
print(df['Gender'].value_counts())

print("\nPreferred Category distribution:")
print(df['Preferred Category'].value_counts())



# age group had a few missing values, so deriving it from age instead
def get_age_group(age):
    if age <= 25:
        return "18-25"
    elif age <= 35:
        return "26-35"
    elif age <= 50:
        return "36-50"
    else:
        return "51+"

df['Age Group'] = df['Age'].apply(get_age_group)
print("\nMissing values after fix:", df['Age Group'].isnull().sum())