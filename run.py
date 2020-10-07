from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


def run(pipeline: Pipeline, parameters, twenty_train, slice=None, twenty_test=None):
    """
    :param pipeline: The pipeline which we will optimize and train
    :param parameters: The parameters which we will be optimizing using GridsearchCV
    :twenty_train: The data we will be using to train the model
    :slice: Only use :slice documents instead of the entire set (roughly 11000 documents)
    """
    data = twenty_train.data[:slice] if slice is not None else twenty_train.data
    target = twenty_train.target[:slice] if slice is not None else twenty_train.target
    length = len(data)
    gridSearcher = GridSearchCV(pipeline, parameters, cv=10, n_jobs=-1)
    # gridsearch only on the first 10% of data to prevent the algorithm from running forever
    # we will refit the best estimator using all data
    results = gridSearcher.fit(data[:length//10], target[:length//10])

    # todo
    # Get the best parameters using gridsearch and use them to train the clf on the entire dataset
    print('KNN with count')
    best_score_small_SVM_vect = gridSearcher.best_score_
    print('best score for KNN vect for dataset: ' +
          str(best_score_small_SVM_vect))

    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, results.best_params_[param_name]))
    # TODO calculate the Precision, recall and F1-score using the testset
    best_estimator = gridSearcher.best_estimator_

    best_estimator.fit(data, target)
    if (twenty_test is not None):
        y_pred = best_estimator.predict(twenty_test.data)
        results = metrics.classification_report(
            twenty_test.targets, y_pred, output_dict=True)['macro avg']
        # macro average / weighted average
        return results

    return None


twenty_train = fetch_20newsgroups(
    subset='train', categories=None, shuffle=True, random_state=42)

# K nearest neighbour pipeline with count
pipeline_clf_KNN_vect = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', MultinomialNB()),
])


# set pipeline parameters
parameters_KNN_vect = {
    #   'vect__ngram_range': [(1, 1), (2, 2)],
    #   'clf__n_neighbors': (1, 2, 3, 4, 5)
}

run(pipeline_clf_KNN_vect, parameters_KNN_vect, twenty_train, slice=4000)
