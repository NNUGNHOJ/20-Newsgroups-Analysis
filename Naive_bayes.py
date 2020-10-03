from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

#create scikit 'bunch' (works like a dict)
twenty_train = fetch_20newsgroups(subset='train', categories=None, shuffle=True, random_state=42)

#Naive Bayes pipeline with count
pipeline_clf_NB_vect = Pipeline([
            ('vect', CountVectorizer()),
            ('clf', MultinomialNB()),
            ])

#Naive Bayes pipeline with transformer
pipeline_clf_NB_tf = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB()),
            ])

#define classifiers data and target
data_clf_NB_vect = pipeline_clf_NB_vect.fit(twenty_train.data, twenty_train.target)
data_clf_NB_tf = pipeline_clf_NB_tf.fit(twenty_train.data, twenty_train.target)

#set pipeline parameters
parameters_NB_vect = {
            'vect__ngram_range': [(1, 1)],
            'clf__alpha': [0.001]
             }

parameters_NB_tf = {
            'vect__ngram_range': [(1, 1)],
            #'vect__lowercase': (True, False),
            #'vect__stop_words': ['english', None],

            #'vect__analyzer': ['word', 'char', 'char_wb'],
            #'vect__max_features': [10, 11, 12, 13, 14, 15, 16,
                                   #17, 18, 19, 20, None],
            'tfidf__use_idf': [(True, False)],
            #'tfidf__smooth_idf': (True, False),
            'clf__alpha': [0.1, 0.01, 0.001, 0.0001]
             }

#tune parameters with 10-fold cross-validation
gs_clf_NB_vect = GridSearchCV(data_clf_NB_vect, parameters_NB_vect, cv=10, n_jobs=-1)
gs_clf_results_NB_vect = gs_clf_NB_vect.fit(twenty_train.data[:800], twenty_train.target[:800])

#display results
print('NB with count')
best_score_small_NB_vect = gs_clf_NB_vect.best_score_
print('best score for NB vect for dataset: ' + str(best_score_small_NB_vect))

for param_name in sorted(parameters_NB_vect.keys()):
        print("%s: %r" % (param_name, gs_clf_results_NB_vect.best_params_[param_name]))

print('count = True')
print('tf = False')

#tune parameters with 10-fold cross-validation
gs_clf_NB_tf = GridSearchCV(data_clf_NB_tf, parameters_NB_tf, cv=10, n_jobs=-1)
clf_results_NB_tf = gs_clf_NB_tf.fit(twenty_train.data[:800], twenty_train.target[:800])

#display results
print('tf with count')
best_score_small_NB_tf = gs_clf_NB_tf.best_score_
print('best score for NB tf for dataset: ' + str(best_score_small_NB_tf))

for param_name in sorted(parameters_NB_tf.keys()):
        print("%s: %r" % (param_name, clf_results_NB_tf.best_params_[param_name]))

print('count = True')
print('tf = True')


