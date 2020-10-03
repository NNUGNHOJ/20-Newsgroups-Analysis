from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV


#create scikit 'bunch' (works like a dict)
twenty_train = fetch_20newsgroups(subset='train', categories=None, shuffle=True, random_state=42)

#Support vector machine pipeline with count
pipeline_clf_SVM_vect = Pipeline([
            ('vect', CountVectorizer()),
            ('clf', SGDClassifier()),
            ])

#Support vector machine pipeline with transformer
pipeline_clf_SVM_tf = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', SGDClassifier()),
            ])

#define classifiers data and target
data_clf_SVM_vect = pipeline_clf_SVM_vect.fit(twenty_train.data, twenty_train.target)
data_clf_SVM_tf = pipeline_clf_SVM_tf.fit(twenty_train.data, twenty_train.target)

#set pipeline parameters
parameters_SVM_vect = {
            'vect__ngram_range': [(1, 1)],
            'clf__loss': ['hinge'],
            'clf__tol': [None],
            'clf__alpha': [0.001]
             }

parameters_SVM_tf = {
            'vect__ngram_range': [(1, 1), (1, 2)],
            #'vect__lowercase': [(1, 1), (1, 2)],
            #'vect__stop_words': ['english', None],
            #'vect__analyzer': ['word', 'char', 'char_wb'],
            #'vect__max_features': [10, 11, 12, 13, 14, 15, 16,
                                   #17, 18, 19, 20, None],
            'tfidf__use_idf': [(True, False)],
            #'tfidf__smooth_idf': (True, False),
            'clf__loss': ['hinge'],
            'clf__tol': [None],
            'clf__alpha': [0.001]
             }

#tune parameters with 10-fold cross-validation
gs_clf_SVM_vect = GridSearchCV(data_clf_SVM_vect, parameters_SVM_vect, cv=10, n_jobs=-1)
gs_clf_results_SVM_vect = gs_clf_SVM_vect.fit(twenty_train.data[:800], twenty_train.target[:800])

#display results
print('SVM with count')
best_score_small_SVM_vect = gs_clf_SVM_vect.best_score_
print('best score for NB vect for dataset: ' + str(best_score_small_SVM_vect))

for param_name in sorted(parameters_SVM_vect.keys()):
        print("%s: %r" % (param_name, gs_clf_results_SVM_vect.best_params_[param_name]))

print('count = True')
print('tf = False')

#tune parameters with 10-fold cross-validation
gs_clf_SVM_tf = GridSearchCV(data_clf_SVM_tf, parameters_SVM_tf, cv=10, n_jobs=-1)
clf_results_SVM_tf = gs_clf_SVM_tf.fit(twenty_train.data[:800], twenty_train.target[:800])

#display results
print('tf with count')
best_score_small_SVM_tf = gs_clf_SVM_tf.best_score_
print('best score for SVM tf for dataset: ' + str(best_score_small_SVM_tf))

for param_name in sorted(parameters_SVM_tf.keys()):
        print("%s: %r" % (param_name, clf_results_SVM_tf.best_params_[param_name]))

print('count = True')
print('tf = True')


