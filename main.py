import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from collections import Counter

#Load in the data
data_train = pd.read_csv('data/train.csv')
data_test = pd.read_csv('data/test.csv')



#Have a quick peek to identify obvious trends in data dirtiness...
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
    df["Fare"].fillna(df["Fare"].mean(), inplace=True)

def replace_incorrect(df):
    '''Replace incorrect values with column mode in dataframe df columns with incorrect categorical values. '''
    other = np.full(df.shape[0], df["Embarked"].mode())
    emb = df["Embarked"]
    emb.where((emb == "S") | (emb == "Q") | (emb == "C"), other=other, inplace=True)

def normalize(df):
    '''Normalized all numerical features to the [0, 1] range. '''
    normalizer = preprocessing.MinMaxScaler()
    for feat in ["Pclass", "Age", "SibSp", "Parch", "Fare"]:
        #Convery the relevant column to a numpy array
        v = df[[feat]].values.astype(float)
        #Normalize
        normalized = normalizer.fit_transform(X=v)
        #Set the data-frame column to the normalized values
        df[feat] = normalized

def add_deck(df):
    '''Add Deck feature, derived from Cabin feature.'''
    #Convert Cabin NaN to NA
    df["Cabin"].fillna("NA", inplace=True)
    #Start the Deck column out with the Cabin values
    decks = df["Cabin"].values
    #Replace each set of cabins with the most common deck
    for i in range(len(decks)):
        if decks[i] == "NA":
            continue
        else:
            decks[i] = Counter([cab[0] for cab in decks[i].split(" ")]).most_common(1)[0][0]

    #Assign new array to datafram
    df["Deck"] = decks

def _parse_title(name):
    for s in name.split(" "):
        if s[len(s) - 1] == ".":
            return s
    return "NA"


def add_title(df):
    '''Add Title feature, derived from Name feature'''
    titles = np.array([_parse_title(name) for name in df["Name"].values])
    df["Title"] = titles


def encode_numeric(df):
    '''Encode string-valued columns as numbers.'''
    number = preprocessing.LabelEncoder()
    for col in ["Name", "Title", "Sex", "Deck", "Embarked"]:
        df[col] = number.fit_transform(df[col])

##Replace NaN's and incorrect values, and normalize
replace_nan(data_train)
replace_nan(data_test)
replace_incorrect(data_train)
replace_incorrect(data_test)
normalize(data_train)
normalize(data_test)

#Add Deck derived feature
add_deck(data_train)
add_deck(data_test)

#Add Title derived feature
add_title(data_train)
add_title(data_test)

##Drop irrelevant columns
data_train.drop(axis=1, labels=["Cabin", "Ticket"], inplace=True)
data_test.drop(axis=1, labels=["Cabin", "Ticket"], inplace=True)

#Label encoding of strings into numeric classes
encode_numeric(data_train)
encode_numeric(data_test)

#Split data into train and test (this is personal testing,
    #Kaggle has given us a testing data-set without labels)
y_train = data_train["Survived"].ravel()
print(y_train.shape)
data_train.drop(axis=1, labels=["Survived"], inplace=True)
X_trn, X_tst, y_trn, y_tst = train_test_split(data_train.values
                                              ,y_train
                                              , test_size=0.33)



#Create and fit a model
logr = LogisticRegression()
logr.fit(X_trn, y_trn)

#Test
print("TEST RESULTS: ")
print(logr.score(X_tst, y_tst))