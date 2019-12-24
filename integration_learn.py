import warnings

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import clone
from sklearn.externals import six
from sklearn.pipeline import _name_estimators

warnings.filterwarnings('ignore')  # 忽略warning


class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    '''
    A majority vote ensemble classifier
    Paramters:
    classifiers:array-like,shape=[n_classifiers],Different classifiers for the ensemble
    vote:str,{'classlabel','probability'},Default:'classlabel',
         if 'classlabel' the prediction is based on the argmax of class labels,
         else if 'probability' ,the argmax of the sum of probabilities is used to predict the class label
         (recommened for calibrated classifier).
    weights:array-like,shape=[n_classifiers],Optional,default:None
         if a list of 'int' or 'float' values are provided,the classifiers are weighted by importance;
         Uses uniform weights if 'weights=None'
    '''

    def __init__(self, classifiers, vote='classlabel', weights=None):
        self.classifiers = classifiers
        self.named_classifiers = {key: value for key, value in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights
        self.lablenc_ = object
        self.classes_ = object
        self.classifiers_ = []

    def fit(self, X, y):
        '''
        Fit classifiers
        Parameters:
        X:{array-like,sparse matrix},shape=[n_samples,n_features],Matrix of training samples.
        y:array-like,shape=[n_sample],Vector of target class labels.
        Returns:
        self:object
        '''
        # use LabelEncoder to ensure class labels start with 0,which is important for np.argmax caoll in self.predict
        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self

    def predict(self, X):
        '''
        Predict class labels for X.
        Parameters:
        X:{array-like,sparse matrix},Shape=[n_samples,n_features],Matrix of testing samples.
        Returns:
        maj_vote:array-like,shape=[n_samples],Predicted class labels.
        '''
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        else:  # 'classlabel' vote
            # collect results from clf.predict Calls
            predictions = np.asarray([clf.predict(X) for clf in self.classifiers_]).T
            maj_vote = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights=self.weights)), axis=1,
                                           arr=predictions)
        maj_vote = self.lablenc_.inverse_transform(maj_vote)
        return maj_vote

    def predict_proba(self, X):
        '''
        Predict class probabilities for X.
        Parameters:
        X:{array-like,sparse matrix},shape={n_samples,n_features},Training vectors,where n_samples is the number of samples and n_features is the number of features.
        Returns:
        avg_proba:array-like,shape=[n_samples,n_classes],Weighted average probability for each class per sample.
        '''
        probas = np.asarray([clf.predict_proba(X) for clf in self.classifiers_])
        avg_proba = np.average(probas, axis=0, weights=self.weights)
        return avg_proba

    def get_params(self, deep=True):
        '''
        Get classifier parameter names for GridSearch.
        '''
        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in six.iteritems(self.named_classifiers):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
            return out


if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    import numpy as np

    iris = datasets.load_iris()
    x, y = iris.data[50:, [1, 2]], iris.target[50:]
    le = LabelEncoder()
    y = le.fit_transform(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=1)
    clf1 = LogisticRegression(penalty="l2", C=0.001, random_state=0)
    clf2 = DecisionTreeClassifier(max_depth=1, criterion="entropy", random_state=0)
    clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric="minkowski")
    pipe1 = Pipeline([["sc", StandardScaler()], ["clf", clf1]])
    pipe3 = Pipeline([["sc", StandardScaler()], ["clf", clf3]])
    clf_labels = ["Logistic Regression", "Decision Tree", "KNN"]
    # for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
    #     scores = cross_val_score(estimator=clf, X=x_train, y=y_train, cv=10, scoring="roc_auc")
    #     print("ROC AUC: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

    mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])
    clf_labels += ["Majority Voting"]
    all_clf = [pipe1, clf2, pipe3, mv_clf]
    for clf, label in zip(all_clf, clf_labels):
        scores = cross_val_score(estimator=clf, X=x_train, y=y_train, cv=10, scoring="roc_auc")
        print("ROC AUC: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

    ####################################################################################################################

    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    colors = ["black", "orange", "blue", "green"]
    line_styles = [":", "--", "-.", "-"]
    for clf, label, clr, ls in zip(all_clf, clf_labels, colors, line_styles):
        y_pred = clf.fit(x_train, y_train).predict_proba(x_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_pred)
        roc_auc = auc(x=fpr, y=tpr)
        plt.plot(fpr, tpr, color=clr, linestyle=ls, label="%s (auc=%0.2f)" % (label, roc_auc))

    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=2)
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.grid()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()
