#Importing all libraries
import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)
import sklearn

df1 = pd.read_csv("https://drive.google.com/uc?id=1TMciVL4jhALM_3uC31oqkgsd2YK6EntK")
df1.shape
df1.columns
df1['AREA'].unique()
df1['AREA'].value_counts()
df1['BUILDTYPE'].value_counts()
df2 = df1.drop(['PRT_ID', 'DIST_MAINROAD','SALE_COND', 'PARK_FACIL','COMMIS', 'UTILITY_AVAIL', 'STREET', 'MZZONE','REG_FEE'
                , 'QS_ROOMS', 'QS_BATHROOM', 'QS_BEDROOM'],axis='columns')
df2.head()
df2.shape
from sklearn.impute import SimpleImputer

# Create a copy of df2 to avoid modifying the original data
df3 = df2.copy()

# Identify numerical columns with missing values
num_cols_with_missing = df3.select_dtypes(include=['number']).columns[df3.select_dtypes(include=['number']).isnull().any()]

# Apply SimpleImputer with mean strategy
imputer = SimpleImputer(strategy='mean')
df3[num_cols_with_missing] = imputer.fit_transform(df3[num_cols_with_missing])

# Check if missing values are filled
(df2.isnull().sum())
df3.isnull().sum()
df3.head()
#Feature Engineering
type(df3['DATE_SALE'][0])
#defining a function to calculate age of property
def calculate_age_build(df, date_sale_col, date_build_col):
    df = df.copy()
    # Convert date columns to datetime 
    df[date_sale_col] = pd.to_datetime(df[date_sale_col], format='%d-%m-%Y')
    df[date_build_col] = pd.to_datetime(df[date_build_col], format='%d-%m-%Y')
    df['AGE_BUILD'] = (df[date_sale_col] - df[date_build_col]).dt.days // 365
    
    return df
df4 = calculate_age_build(df3, 'DATE_SALE', 'DATE_BUILD')
df4.head()
df4['N_BEDROOM'].unique()
df4['AGE_BUILD'].unique()
df4['N_BATHROOM'].unique()
df4['N_ROOM'].unique()

df5 = df4.drop(['DATE_SALE' , 'DATE_BUILD'],axis = 'columns')
df5.head()
df5['AREA'].unique()
df5['AREA'] = df5['AREA'].replace(
    ['TNagar', 'Chrompt', 'Chrmpet', 'Karapakam', 'Ana Nagar', 'Chormpet', 
     'Adyr', 'Velchery', 'Ann Nagar', 'KKNagar'],
    ['T Nagar', 'Chrompet', 'Chrompet', 'Karapakkam', 'Anna Nagar', 'Chrompet', 
     'Adyar', 'Velachery', 'Anna Nagar', 'KK Nagar']
)
df5['AREA'].unique()
df5.head(10)
df5['INT_SQFT'].unique()

#plotting graph

plt.figure(figsize=(10, 6))
plt.scatter(df5['INT_SQFT'], df5['N_ROOM'], color='blue')
plt.title('INT_SQFT vs N_ROOM', fontsize=14)
plt.xlabel('INT_SQFT (Square Feet)', fontsize=12)
plt.ylabel('N_ROOM (Number of Rooms)', fontsize=12)
plt.show()


df5 = df5[~((df5['INT_SQFT'] >= 2000) & (df5['INT_SQFT'] <= 2100) & (df5['N_ROOM'] == 6))]
df5.head()

plt.figure(figsize=(10, 6))
plt.scatter(df5['INT_SQFT'], df5['N_ROOM'], color='blue')
plt.title('INT_SQFT vs N_ROOM', fontsize=14)
plt.xlabel('INT_SQFT (Square Feet)', fontsize=12)
plt.ylabel('N_ROOM (Number of Rooms)', fontsize=12)
plt.show()
df5['AGE_BUILD'].unique()
df5['AREA'].unique()
df5.head()
df5['BUILDTYPE'].value_counts()
df5['BUILDTYPE'] = df5['BUILDTYPE'].replace(
    ['Comercial','Other'],
    ['Commercial','Others']
)
df5['BUILDTYPE'].value_counts()
dummies = pd.get_dummies(df5.AREA)
dummies.head()

df6 = pd.concat([df5,dummies], axis = 'columns')
df6.head()
dummies1 = pd.get_dummies(df5.BUILDTYPE)
dummies1.head()

df7 = pd.concat([df6,dummies1], axis = 'columns')
df7.head()
df8 = df7.drop('BUILDTYPE',axis='columns')
df8.head()
df9 = df8.drop('AREA',axis='columns')
df9.head()
df9.shape

X = df9.drop('SALES_PRICE',axis='columns')
X.head()
y = df9.SALES_PRICE
y.head()

#Building a linear regression model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)
from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test) #calculates r2 score

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

score = cross_val_score(LinearRegression(), X, y, cv=cv)
print("R^2 score : ", score)
# Predictions
y_pred = lr_clf.predict(X_test) 

# Display results
print("Training Data (X_train):")
print(X_train.head())
print("\nTraining Labels (y_train):")
print(y_train.head())
print("\nTesting Data (X_test):")
print(X_test.head())
print("\nTesting Labels (y_test):")
print(y_test.head())
print("\nPredicted Prices:")
print(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).head())

