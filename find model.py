#Importing libraries for initial exploratory analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mlt
from sklearn.svm import SVC

#For Feature Engineering:


#For Predictive Modeling
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#For Model Evaluation
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import plot_confusion_matrix, accuracy_score, roc_curve, auc, log_loss
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

import joblib

pd.set_option("display.max_rows",None)
df = pd.read_csv('heart.csv')

print(df.head())

print(df.tail())
print(df.shape)
print(df.size)
print(df.info())
df.ExerciseAngina = df.ExerciseAngina.replace(to_replace = ['Y','N'],value = [1,0])
df.Sex = df.Sex.replace(to_replace = ['F','M'],value = [1,0])
print(df.ChestPainType.unique())
df.ChestPainType = df.ChestPainType.replace(to_replace = ['ATA','NAP','ASY','TA'], value = [1,2,3,4])
#df.RestingECG = df.RestingECG(to_replace = ['Normal','ST'], value = [1,0])

restingesg_mapping = {k:v for v, k in enumerate(df.RestingECG.unique())}
df['RestingECG'] = df['RestingECG'].map(restingesg_mapping)



ST_Slope = {k:v for v, k in enumerate(df.ST_Slope.unique())}
df['ST_Slope'] = df['ST_Slope'].map(ST_Slope)
#df.ExerciseAngina = df.ExerciseAngina(to_replace = ['Y','N'], value = (1,0))
# visulazaiton dataset
#graph the pairplot across variables which might effect the chances of Heart Disease
sns.pairplot(df,hue="HeartDisease")
plt.savefig('pairplot.pdf')
plt.show()
plt.close()
## from the graph we can tell that the heart disease seems to be distincitve amongst the peeopl with:
# 1.more common heart diseaes happen amongs elderly age howerver
# 2. the people with a higehr heart Rate(aka MAxHR)
# 3.it's a higher chance for poeple who have high sugar compared to low.
#4.high age with high cholesterol
#5 people of lower age than 45 with high hr have a low probaility of getting a heart disease


#chek data qualkity( standard deivation, mean value.... to understanding what kinda of data we are working with
print(df.describe())
print(df['HeartDisease'].value_counts())

#visulization
cor = df.corr()
plt.figure(figsize=(7,7))
sns.heatmap(cor, annot=True)
plt.title('correlation')
plt.savefig('correlation.pdf')
plt.show()
plt.close()
## from the the correlation we can see that HaxHR and Oldpeak has a highly correlated with most of features.
##split data into feature train and featire test
X = df.drop('HeartDisease', axis=1)
Y = df[['HeartDisease']]

x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size=0.2,random_state=123)
print("X_train shape:", x_train.shape, "X_test shape:", x_test.shape,"y_train shape:", y_train.shape,"y_test shape:", y_test.shape)
#logistic regression model


fit1 = LogisticRegression(solver='liblinear').fit(x_train,y_train)
fit2 = LogisticRegression(penalty = 'l1', solver='liblinear').fit(x_train,y_train)
fit3 = LogisticRegression(penalty = 'l2', solver='liblinear').fit(x_train,y_train)

y_pred1 = fit1.predict(x_test)
y_pred2 = fit2.predict(x_test)
y_pred3 = fit3.predict(x_test)

y_pred_proba1 = fit1.predict_proba(x_test)[:, 1]

y_pred_proba2 = fit2.predict_proba(x_test)[:, 1]

y_pred_proba3 = fit3.predict_proba(x_test)[:, 1]

[fpr1, tpr1, thr1] = roc_curve(y_test, y_pred_proba1)
[fpr2, tpr2, thr2] = roc_curve(y_test, y_pred_proba2)
[fpr3, tpr3, thr3] = roc_curve(y_test, y_pred_proba1)
# result for logistic regression 1 model
print('Train/Test split results:')
print("LogisticRegression1"+" accuracy is %2.3f" % accuracy_score(y_test, y_pred1))
print("LogisticRegression1"+" log_loss is %2.3f" % log_loss(y_test, y_pred_proba1))
print("LogisticRegression1"+" auc is %2.3f" % auc(fpr1, tpr1))

# result for logistic regression 2 model
print('Train/Test split results:')
print("LogisticRegression2"+" accuracy is %2.3f" % accuracy_score(y_test, y_pred2))
print("LogisticRegression2"+" log_loss is %2.3f" % log_loss(y_test, y_pred_proba2))
print("LogisticRegression2"+" auc is %2.3f" % auc(fpr2, tpr2))


# result for logistic regression 3 model
print('Train/Test split results:')
print("LogisticRegression3"+" accuracy is %2.3f" % accuracy_score(y_test, y_pred3))
print("LogisticRegression3"+" log_loss is %2.3f" % log_loss(y_test, y_pred_proba3))
print("LogisticRegression3"+" auc is %2.3f" % auc(fpr3, tpr3))


idx = np.min(np.where(tpr1 > 0.95)) # index of the first threshold for which the sensibility > 0.95
#grpah for logistic regression 1 model
plt.figure()
plt.plot(fpr1, tpr1, color='coral', label='ROC curve (area = %0.3f)' % auc(fpr1, tpr1))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot([0,fpr1[idx]], [tpr1[idx],tpr1[idx]], 'k--', color='blue')
plt.plot([fpr1[idx],fpr1[idx]], [0,tpr1[idx]], 'k--', color='blue')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - specificity)', fontsize=14)
plt.ylabel('True Positive Rate (recall)', fontsize=14)
plt.title('(ROC) curve1')
plt.legend(loc="lower right")

plt.savefig('LR1.jpg')
plt.show()
plt.close()
#grpah for logistic regression 2 model
plt.figure()
plt.plot(fpr2, tpr2, color='coral', label='ROC curve (area = %0.3f)' % auc(fpr2, tpr2))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot([0,fpr2[idx]], [tpr2[idx],tpr2[idx]], 'k--', color='blue')
plt.plot([fpr2[idx],fpr2[idx]], [0,tpr2[idx]], 'k--', color='blue')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - specificity)', fontsize=14)
plt.ylabel('True Positive Rate (recall)', fontsize=14)
plt.title('(ROC) curve2')
plt.legend(loc="lower right")

plt.savefig('LR2.jpg')
plt.show()
plt.close()
#graph for logistic regression 3 model

plt.figure()
plt.plot(fpr3, tpr3, color='coral', label='ROC curve (area = %0.3f)' % auc(fpr3, tpr3))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot([0,fpr3[idx]], [tpr3[idx],tpr3[idx]], 'k--', color='blue')
plt.plot([fpr3[idx],fpr1[idx]], [0,tpr3[idx]], 'k--', color='blue')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - specificity)', fontsize=14)
plt.ylabel('True Positive Rate (recall)', fontsize=14)
plt.title('(ROC) curve3')
plt.legend(loc="lower right")

plt.savefig('LR3.jpg')
plt.show()
plt.close()
#SVM

# instantiate linear kernel svm classifier with default hyperparameters
clf1=SVC(random_state=42, kernel='linear',probability=True)

# fit classifier to training set

svmfit = clf1.fit(x_train,y_train)

# make predictions on test set
y_pred=clf1.predict(x_test)
# compute and print accuracy score round 4
print('linear svm accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
# Evaluate

plot_confusion_matrix(svmfit , x_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize='true')
plt.title('Confusion matrix for RBF SVM')


plt.savefig('linearSVM.jpg')
plt.show()
plt.close()
#instantiate linear radial svm classifier with default hyperparameters

clf2=SVC(random_state=42, kernel='rbf')

# fit classifier to training set

svmfit2 = clf2.fit(x_train,y_train)

# make predictions on test set
y_pred=clf1.predict(x_test)
# compute and print accuracy score round 4
print('radial svm accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
# Evaluate
plot_confusion_matrix(svmfit2 , x_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize='true')
plt.title('Confusion matrix for RBF SVM')

plt.savefig('RadilSVM.jpg')
plt.show()
plt.close()
## LDA

# fit model
model = LinearDiscriminantAnalysis()

ldamodel = model.fit(x_train,y_train)

y_predlda = ldamodel.predict(x_test)

y_pred_probalda = ldamodel.predict_proba(x_test)[:, 1]

[fprlda, tprlda, thrlda] = roc_curve(y_test, y_pred_probalda)
# result for LDA model
print('Train/Test split results:')
print("LDA"+" accuracy is %2.3f" % accuracy_score(y_test, y_predlda))
print("LDA"+" log_loss is %2.3f" % log_loss(y_test, y_pred_probalda))
print("LDA"+" auc is %2.3f" % auc(fprlda, tprlda))

idx = np.min(np.where(tpr1 > 0.95)) # index of the first threshold for which the sensibility > 0.95
#grpah for LDA model
plt.figure()
plt.plot(fprlda, tprlda, color='coral', label='ROC curve (area = %0.3f)' % auc(fprlda, tprlda))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot([0,fprlda[idx]], [tprlda[idx],tprlda[idx]], 'k--', color='blue')
plt.plot([fprlda[idx],fprlda[idx]], [0,tprlda[idx]], 'k--', color='blue')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - specificity)', fontsize=14)
plt.ylabel('True Positive Rate (recall)', fontsize=14)
plt.title('(ROC) curve LDA')
plt.legend(loc="lower right")

plt.savefig('LDA.jpg')
plt.show()
plt.close()


## linear svm wit default hyperparemeters has the best accuracy among all the prediciton model

filename = 'svm_model.sav'
joblib.dump(svmfit, filename)