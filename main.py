import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from collections import Counter

#Load in the data
data_train = pd.read_csv('data/train.csv')
data_test = pd.read_csv('data/test.csv')



#Have a quick peek to identify obvious trends in data dirtiness...
print(data_train["Name"])
print(data_train.describe())
print(data_train.head(20))
print(data_train.isnull().sum()) ##This indicates Age and Cabin are particularly bad offenders
                            ##for missing values

#Identify incorrect values from features with a finite discrete value set...
print(data_train[~data_train["Embarked"].isin(["C", "Q", "S"])].shape) ##This indicates 2 rows have incorrect values for the "Embarked" field
print(data_train[~data_train["Pclass"].isin([1, 2, 3])].shape)
print(data_train[~data_train["Sex"].isin(["male", "female"])].shape)
print(data_train[~data_train["Survived"].isin([0, 1])].shape)

#We can skip outlier identification for most columns. Looking at the data description, min/max values for each column
#don't suggest anything too crazy.
#Just to be safe, we'll have a look at the Fare column where the max value seems quite far from the mean and
#could be an incorrect measurement.
print(data_train[data_train["Fare"] > 500].shape)
data_train["Fare"].plot(kind='hist', title='Fare')
plt.show() #This suggests bins of ticket prices that reflect degrees of luxury.

#Enter the pipeline...
def replace_nan(df):
    '''Replace NaN values in columns with values respecitvely.
     - Age with Column Mean
     - Embarked with Column Mode
     - Fare with Column Mean
     '''
    df["Age"].fillna(df["Age"].mean(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode(), inplace=True)
    df["Fare"].fillna(df["fare"].mean(), inplace=True)

def normalize(df):
    '''Normalized all numerical features to the [0, 1] range. '''
    normalizer = preprocessing.MinMaxScaler
    for feat in ["Pclass", "Age", "Sibsp", "Parch", "Fare"]:
        #Convery the relevant column to a numpy array
        v = df[[feat]].values.astype(float)
        #Normalize
        normalized = normalizer.fit_transform(v)
        #Set the data-frame column to the normalized values
        df[feat] = normalized

def add_deck(df):
    '''Add Deck feature, derived from Cabin feature.'''
    decks = df["Cabin"].values()
    #Convert NaNs to NA
    decks[math.isnan(decks)] = "NA"
    #Take the most common cabin in each passenger's purchased set of cabins
    decks = np.array([Counter([deck[0] for deck in deck_set.split(" ")]).most_common(1)[0][0] for deck_set in decks])
    #Assign new array to datafram
    df = df.assign(Deck=decks)

def _parse_title(name):
    for s in name.split(" "):
        if s[len(s) - 1] == ".":
            return s
    return "NA"


def add_title(df):
    '''Add Title feature, derived from Name feature'''
    titles = np.array([_parse_title(name) for name in df["Name"].values()])
    df.assign(Title=titles)

##Replace NaN's and normalize
replace_nan(data_train)
replace_nan(data_test)
normalize(data_train)
normalize(data_test)

#Add Deck derived feature
add_deck(data_train)
add_deck(data_test)

#Add Title derived feature
add_title(data_train)
add_title(data_test)

##Drop irrelevant columns
data_train.drop(columns=["Cabin", "Ticket"])
data_test.drop(columns=["Cabin", "Ticket"])


