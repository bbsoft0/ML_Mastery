# Machine Learning Mastery + Kaggle

Machine Learning Mastery - complete courses python files - combined with Kaggle Courses jupyter files.

Kaggle is the most popular platform for Data Science. It has multiple free datasets, projects that you can use for practice, and competitions. It also has a helpful community where you can share your thoughts and learn new things. But the best feature of Kaggle is Kaggle Learn. Even if you don’t know anything about data science, you can learn all the basics from Kaggle Courses and then move on to sharpening your skills by doing projects.

In this repository, the Kaggle learn course tutorial and excerses nootbooks(.ipynb) are available which I have done and earned completion certificates.
The Kaggle datasets are avalable in the inputKaggle folder.
The Mastery datasets are avalable in the inputMastery folder.
The courses structure is as follows:

# Complete Courses

<!-- TOC -->

- [Kaggle Courses](#kaggle)
- [Machine Learning Mastery](#machine-learning-mastery)

<!-- /TOC -->

## Kaggle

P01. [Python Basics](./P01_Python_Basics.py)  
Functions, Lists, Strings and Dictionaries

P02. [Guessing](./P02_Guessing.py)  
Guessing the number - User has to guess the number picked by computer

P03. [Age](./P03_Age.py)  
User introduces age (can be decimal) and gets the age in seconds

P04. [PriceOfAChair](./P04_PriceOfAChair.py)  
BeautifulSoup used to download a page and then individual data is obtained from the page

P05. [RandomNumbers](./P05_RandomNumbers.py)  
Uses numpy pseudorandom number generator to generate random numbers between 1...45, 1...20

P06. [Dictionary](./P06_Dictionary.py)  
Interactive dictionary - uses data.json and displays information about the words introduced.

K02. [Intro To Machine Learning](./K02_introML.ipynb)  
 Starts with DecisionTrees then gets to RandomForest which has the best performance

K03. [Pandas](./K03_pandas.ipynb)  
 Uses pandas to read Wine data, describe it, fillna and work with columns

K04_z0. [Intermediate Machine Learning](./K04_z0_intermediateML.ipynb)  
Uses 4 RandomForest models from point 2 to train , find best model then generate a submission

K04_z1. [Housing Prices Competition](./K04_z1_HousingPricesCompetition.ipynb)  
Compare DecisionTreeRegressor with RandomForest model - the best is RandomForest then generate a submission

K04_z2. [Pipelines](./K04_z2_pipeline.ipynb)  
Pipelines are a simple way to keep your data preprocessing and modeling code organized. Specifically, a pipeline bundles preprocessing and modeling steps so you can use the whole bundle as if it were a single step.

K04_z3. [XGBoost](./K04_z3_xgboost.ipynb)  
Gradient boosting is a method that goes through cycles to iteratively add models into an ensemble. We use the loss function to fit a new model that will be added to the ensemble. Specifically, we determine model parameters so that adding this new model to the ensemble will reduce the loss.

K05_z0. [Data Vizualization](./K05_z0_data_visualization.ipynb)  
 Seaborn, Line Charts, Custom Styles, Heat Maps

K05_z1. [Breast Cancer Detection](./K05_z1_breast_cancer_detection.ipynb)  
Histograms for benign and maligant tumors, KDE plots

K06. [Feature Engineering](./K06_feature_engineering.ipynb)  
Features, Clustering with K-Means, Principal Component Analysis

K07. [Data Cleaning](./K07_data_cleaning.ipynb)  
Minmax_scaling, Normalization, Remove trailing white spaces, fuzzywuzzy closest match

K08. [Intro to Deep Learning](./K08_introDeepLearning.ipynb)  
Activation Layer, relu, Plot

K09. [KerasGradient](./K09_KerasGradient.ipynb)  
Preprocessor, Transformer, Added loss and optimizer, Plot

K10. [KerasUnderfitOverfit](./K10_KerasUnderfitOverfit.ipynb)  
Do a "Grouped" split to keep all of an artist's songs in one split or the other - prevents signal leakage. Simple Network - linear model underfit. Added three hidden layers - overfit. Added early stopping callback.

K11. [BinaryClassification](./K11_BinaryClassification.ipynb)  
 In Regression, MAE = distance between the expected outcome and the predicted outcome. In Classification Cross-Entropy = distance between probabilities. Sigmoid activation - covert the real-valued outputs produced by a dense layer into probabilities.

## Machine Learning Mastery

001\_. [Probability](./001_Probability.py)  
 Gaussian Distribution, Bayes, cross entropy H(P, Q), Naive classifier, Log Loss, Brier score
