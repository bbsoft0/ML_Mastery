# Data Preparation for Machine Learning

# Lesson 02: Fill Missing Values With Imputation
# statistical imputation transform for the horse colic dataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
from sklearn.datasets import make_classification
from numpy import isnan
from pandas import read_csv
from sklearn.impute import SimpleImputer
# load dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/horse-colic.csv'
dataframe = read_csv(url, header=None, na_values='?')
# split into input and output elements
data = dataframe.values
ix = [i for i in range(data.shape[1]) if i != 23]
X, y = data[:, ix], data[:, 23]
# print total missing
print('Missing: %d' % sum(isnan(X).flatten()))
# define imputer
imputer = SimpleImputer(strategy='mean')
# fit on the dataset
imputer.fit(X)
# transform the dataset
Xtrans = imputer.transform(X)
# print total missing
print('Missing: %d' % sum(isnan(Xtrans).flatten()))


# Lesson 03: Select Features With RFE
# report which features were selected by RFE
# define dataset
X, y = make_classification(
    n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
# define RFE
rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=5)
# fit RFE
rfe.fit(X, y)
# summarize all features
for i in range(X.shape[1]):
    print('Column: %d, Selected=%s, Rank: %d' %
          (i, rfe.support_[i], rfe.ranking_[i]))

# Lesson 04: Scale Data With Normalization
# example of normalizing input data
# define dataset
X, y = make_classification(
    n_samples=1000, n_features=5, n_informative=5, n_redundant=0, random_state=1)
# summarize data before the transform
print("Summarize data before the transform")
print(X[:3, :])
# define the scaler
trans = MinMaxScaler()
# transform the data
X_norm = trans.fit_transform(X)
# summarize data after the transform
print("Summarize data after the transform")
print(X_norm[:3, :])

# Lesson 05: Transform Categories With One-Hot Encoding
# one-hot encode the breast cancer dataset
# define the location of the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/breast-cancer.csv"
# load the dataset
dataset = read_csv(url, header=None)
# retrieve the array of data
data = dataset.values
# separate into input and output columns
X = data[:, :-1].astype(str)
y = data[:, -1].astype(str)
# summarize the raw data
print("Encode categorical input variables as numbers. Initial:")
print(X[:3, :])
# define the one hot encoding transform
encoder = OneHotEncoder(sparse=False)
# fit and apply the transform to the input data
X_oe = encoder.fit_transform(X)
# summarize the transformed data
print("Summarize the transformed data")
print(X_oe[:3, :])


# Lesson 06: Transform Numbers to Categories With kBins
# discretize numeric input variables
# define dataset
X, y = make_classification(
    n_samples=1000, n_features=5, n_informative=5, n_redundant=0, random_state=1)
# summarize data before the transform
print("Summarize data before the transform")
print(X[:3, :])
# define the transform
trans = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
# transform the data
X_discrete = trans.fit_transform(X)
# summarize data after the transform
print("Summarize data after the transform")
print(X_discrete[:3, :])


# Lesson 07: Dimensionality Reduction With PCA
# example of pca for dimensionality reduction
# define dataset
X, y = make_classification(
    n_samples=1000, n_features=10, n_informative=3, n_redundant=7, random_state=1)
# summarize data before the transform
print("Summarize data before the transform")
print(X[:3, :])
# define the transform
trans = PCA(n_components=3)
# transform the data
X_dim = trans.fit_transform(X)
# summarize data after the transform
print("Summarize data after the transform")
print(X_dim[:3, :])
