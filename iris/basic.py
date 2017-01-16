# ref: http://machinelearningmastery.com/machine-learning-in-python-step-by-step/

# Load libraries
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


def section(name):
    print("\n------{}------".format(name))


def sub_section(name):
    print("\n--------{}----".format(name))


# wget https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
section("loading data")
url = "../data/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

# shape
section('dimensions of dataset')
print(dataset.shape)

# head
section('peak at(eyeball) the data')
print(dataset.head(20))

section('statistical summary')
print(dataset.describe())

section('class distribution')
print(dataset.groupby('class').size())

# Given that the input variables are numeric, we can create box and whisker plots of each.
section('data visualization - univariate')
dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
plt.show()

section('data visualization - histograms')
dataset.hist()
plt.show()
# it looks like perhaps two of the input variables have a Gaussian distribution. This is useful to note as we can use algorithms that can
#  exploit this assumption.


section('data visualization - multivariate plots')
scatter_matrix(dataset)
plt.show()
# Note the diagonal grouping of some pairs of attributes. This suggests a high correlation and a predictable relationship.

section('evaluate algorithms')

# 1. Separate out a validation dataset.
# 2. Set-up the test harness to use 10-fold cross validation.
# 3. Build 5 different models to predict species from flower measurements
# 4. Select the best model.

sub_section('create a validation dataset')
array = dataset.values
X = array[:, 0:4]
Y = array[:, 4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

sub_section('test harness')
seed = 7
scoring = 'accuracy'

sub_section('building models')
models = [
    ('LR', LogisticRegression()),
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier()),
    ('NB', GaussianNB()),
    ('SVM', SVC())
]

sub_section('evaluating each model')
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

sub_section('select best model')
# We now have 6 models and accuracy estimations for each.
# LR: 0.966667 (0.040825)
# LDA: 0.975000 (0.038188)
# KNN: 0.983333 (0.033333)
# CART: 0.975000 (0.038188)
# NB: 0.975000 (0.053359)
# SVM: 0.981667 (0.025000)

# KNN has the largest estimated accuracy score.

# We can also create a plot of the model evaluation results and compare the spread and the mean accuracy of each model. There is a
# population of accuracy measures for each algorithm because each algorithm was evaluated 10 times (10 fold cross validation).

sub_section('comparing algorithms')

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# You can see that the box and whisker plots are squashed at the top of the range, with many samples achieving 100% accuracy.

section('make predictions')
# The KNN algorithm was the most accurate model that we tested. Now we want to get an idea of the accuracy of the model on our validation
#  set.

# This will give us an independent final check on the accuracy of the best model. It is valuable to keep a validation set just in case
# you made a slip during training, such as overfitting to the training set or a data leak. Both will result in an overly optimistic result.

# We can run the KNN model directly on the validation set and summarize the results as a final accuracy score, a confusion matrix and a
# classification report.
sub_section('make predictions on validation dataset')
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
# model.predict() to predict the new data
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# We can see that the accuracy is 0.9 or 90%. The confusion matrix provides an indication of the three errors made. Finally the
# classification report provides a breakdown of each class by precision, recall, f1-score and support showing excellent results (granted
# the validation dataset was small).
