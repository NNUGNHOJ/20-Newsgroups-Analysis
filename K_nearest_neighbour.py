from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

#create scikit 'bunch' (works like a dict)
twenty_train = fetch_20newsgroups(subset='train', categories=None, shuffle=True, random_state=42)

#K nearest neighbour pipeline with count
pipeline_clf_KNN_vect = Pipeline([
            ('vect', CountVectorizer()),
            ('clf', KNeighborsClassifier()),
            ])

#K nearest neighbour pipeline with transformer
pipeline_clf_KNN_tf = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', KNeighborsClassifier()),
            ])

#define classifiers data and target
data_clf_KNN_vect = pipeline_clf_KNN_vect.fit(twenty_train.data, twenty_train.target)
data_clf_KNN_tf = pipeline_clf_KNN_tf.fit(twenty_train.data, twenty_train.target)

#set pipeline parameters
parameters_KNN_vect = {
            'vect__ngram_range': [(1, 1)],
            'clf__n_neighbors': (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
             }

parameters_KNN_tf = {
            'vect__ngram_range': [(1, 1)],
            #'vect__lowercase': [(1, 1), (1, 2)],
            #'vect__stop_words': ['english', None],
            #'vect__analyzer': ['word', 'char', 'char_wb'],
            #'vect__max_features': [10, 11, 12, 13, 14, 15, 16,
                                   #17, 18, 19, 20, None],
            'tfidf__use_idf': [(True, False)],
            #'tfidf__smooth_idf': (True, False),
            'clf__n_neighbors': (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
             }

#tune parameters with 10-fold cross-validation
gs_clf_KNN_vect = GridSearchCV(data_clf_KNN_vect, parameters_KNN_vect, cv=10, n_jobs=-1)
gs_clf_results_KNN_vect = gs_clf_KNN_vect.fit(twenty_train.data[:800], twenty_train.target[:800])

#display results
print('KNN with count')
best_score_small_SVM_vect = gs_clf_KNN_vect.best_score_
print('best score for KNN vect for dataset: ' + str(best_score_small_SVM_vect))

for param_name in sorted(parameters_KNN_vect.keys()):
        print("%s: %r" % (param_name, gs_clf_results_KNN_vect.best_params_[param_name]))

print('count = True')
print('tf = False')

#tune parameters with 10-fold cross-validation
gs_clf_KNN_tf = GridSearchCV(data_clf_KNN_tf, parameters_KNN_tf, cv=10, n_jobs=-1)
clf_results_KNN_tf = gs_clf_KNN_tf.fit(twenty_train.data[:800], twenty_train.target[:800])

#display results
print('tf with count')
best_score_small_SVM_tf = gs_clf_KNN_tf.best_score_
print('best score for KNN tf for dataset: ' + str(best_score_small_SVM_tf))

for param_name in sorted(parameters_KNN_tf.keys()):
        print("%s: %r" % (param_name, clf_results_KNN_tf.best_params_[param_name]))

print('count = True')
print('tf = True')


