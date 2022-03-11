# Randomly sample Gaussian Distribution
from sklearn.metrics import brier_score_loss
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
from numpy import asarray
from math import log2
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_blobs
from numpy.random import normal
# define the distribution
mu = 50
sigma = 5
n = 10
# generate the sample
sample = normal(mu, sigma, n)
print("Randomly sample Gaussian Distribution")
print(sample)


# example of gaussian naive bayes
# generate 2d classification dataset
print("Gaussian naive bayes")
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
# define the model
model = GaussianNB()
# fit the model
model.fit(X, y)
# select a single sample
Xsample, ysample = [X[0]], y[0]
# make a probabilistic prediction
yhat_prob = model.predict_proba(Xsample)
print('Predicted Probabilities: ', yhat_prob)
# make a classification prediction
yhat_class = model.predict(Xsample)
print('Predicted Class: ', yhat_class)
print('Truth: y=%d' % ysample)


# example of calculating cross entropy
# calculate cross entropy
print("Cross entropy")


def cross_entropy(p, q):
    return -sum([p[i]*log2(q[i]) for i in range(len(p))])


# define data
p = [0.10, 0.40, 0.50]
q = [0.80, 0.15, 0.05]
# calculate cross entropy H(P, Q)
ce_pq = cross_entropy(p, q)
print('H(P, Q): %.3f bits' % ce_pq)
# calculate cross entropy H(Q, P)
ce_qp = cross_entropy(q, p)
print('H(Q, P): %.3f bits' % ce_qp)


# example of the majority class naive classifier in scikit-learn
# define dataset
print("Naive classifier")
X = asarray([0 for _ in range(100)])
class0 = [0 for _ in range(25)]
class1 = [1 for _ in range(75)]
y = asarray(class0 + class1)
# reshape data for sklearn
X = X.reshape((len(X), 1))
# define model
model = DummyClassifier(strategy='most_frequent')
# fit model
model.fit(X, y)
# make predictions
yhat = model.predict(X)
# calculate accuracy
accuracy = accuracy_score(y, yhat)
print('Accuracy: %.3f' % accuracy)

# example of log loss
# define data
y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
y_pred = [0.8, 0.9, 0.9, 0.6, 0.8, 0.1, 0.4, 0.2, 0.1, 0.3]
# define data as expected, e.g. probability for each event {0, 1}
y_true = asarray([[v, 1-v] for v in y_true])
y_pred = asarray([[v, 1-v] for v in y_pred])
# calculate log loss
loss = log_loss(y_true, y_pred)
print('Log Loss: %.3f' % loss)

# example of brier loss
# define data
y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
y_pred = [0.8, 0.9, 0.9, 0.6, 0.8, 0.1, 0.4, 0.2, 0.1, 0.3]
# calculate brier score
score = brier_score_loss(y_true, y_pred, pos_label=1)
print('Brier score: %.3f' % score)
