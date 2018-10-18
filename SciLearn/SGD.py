from sklearn.linear_model import SGDClassifier


from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc



from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
breast_cancer = datasets.load_breast_cancer()

x_train, x_test, y_train, y_test = \
    train_test_split(breast_cancer.data, breast_cancer.target, test_size=0.1)

clf = linear_model.SGDClassifier(max_iter=10000, loss='log')
logisticclf = linear_model.LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')

clf.fit(x_train, y_train)
logisticclf.fit(x_train, y_train)

y_predict = clf.predict(x_test)
y_predict_prob = clf.predict_proba(x_test)
y_predict_prob_lr = logisticclf.predict_proba(x_test)

fpr, tpr, thresholds = roc_curve(y_test, y_predict_prob[:, 1])
roc_auc = auc(fpr, tpr)
print roc_auc


fpr, tpr, thresholds = roc_curve(y_test, y_predict_prob_lr[:, 1])
roc_auc = auc(fpr, tpr)

#print "haha"
print roc_auc
plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold')
plt.show()

#print clf.coef_
#print y_train
#print y_test
#print y_predict
#print y_predict_prob


#print breast_cancer




