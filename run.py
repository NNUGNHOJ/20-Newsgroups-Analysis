from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn import metrics


def run(pipeline: Pipeline, twenty_train, twenty_test, slice=None):
    """
    Runs a single 'experiment' (given a pipeline, fit the classifier and calculate the precision, recall, F1-score)
    :param pipeline: The pipeline which we will optimize and train
    :param parameters: The parameters which we will be optimizing using GridsearchCV
    :twenty_train: The data we will be using to train the model
    :slice: Only use :slice documents instead of the entire set (roughly 11000 documents)
    """
    data = twenty_train.data[:slice] if slice is not None else twenty_train.data
    target = twenty_train.target[:slice] if slice is not None else twenty_train.target

    pipeline.fit(data, target)

    y_pred = pipeline.predict(twenty_test.data)
    results = metrics.classification_report(
        twenty_test.target, y_pred, output_dict=True, digits=2)
    # macro average / weighted average
    return results


def runOnAllCombinations():
    """
    Runs all combinations we need (3 classifiers x the 3 feature-variants)
    Gathers the resulting classification_reports and returns them as a list
    """
    twenty_train = fetch_20newsgroups(
        subset='train', categories=None, shuffle=True, random_state=42)
    twenty_test = fetch_20newsgroups(
        subset='test', categories=None, shuffle=True, random_state=42)

    features = [
        [('vect', CountVectorizer())],
        [('vect', CountVectorizer()), ('tfidf', TfidfTransformer(use_idf=False))],
        [('vect', CountVectorizer()), ('tfidf', TfidfTransformer(use_idf=True))],
    ]
    classifiers = [
        [('clf', MultinomialNB())],
        [('clf', KNeighborsClassifier(n_neighbors=1))],
        [('clf', SGDClassifier())],
    ]

    pipelines = [Pipeline(feature + classifier)
                 for classifier in classifiers for feature in features]

    classification_reports = []
    for pipeline in pipelines:
        classification_reports.append(run(pipeline, twenty_train, twenty_test))

    return classification_reports

# uncomment this line to run the entire experiment
# clf_reports = runOnAllCombinations()

# after running the above code, we find that the best model is
# the SGDClassifier using all features.

# experiment using the SGDClassifier


def experiment():
    twenty_train = fetch_20newsgroups(
        subset='train', categories=None, shuffle=True, random_state=42)
    twenty_test = fetch_20newsgroups(
        subset='test', categories=None, shuffle=True, random_state=42)

    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer(use_idf=True)),
        ('clf', SGDClassifier()),
    ])

    # we used grid search to try to find an improvent over the 'vanilla' score
    # that we obtained earlier (above)
    params1 = {
        'vect__lowercase': [True],
        # TODO do we want to provide another list?
        'vect__stop_words': [['the'], None],
        'vect__max_features': range(130000, 131000, 100),
        # TODO also add gridsearch for the analyzer + ngram
    }

    params2 = {
        'vect__analyzer': ['char', ],
        # TODO do we want to provide another list?
        'vect__ngram_range': [(5, 5), (4, 6)],
        # TODO also add gridsearch for the analyzer + ngram
    }
    # best params
    #

    gridSearcher = GridSearchCV(pipeline, params2, n_jobs=-1)
    gridSearcher.fit(twenty_train.data, twenty_train.target)
    # best params
    # lc=true
    # stopwords=english
    # max_features=120000
    return gridSearcher


def printClassificationReport(model, twenty_test):
    y_pred = model.predict(twenty_test.data)
    results = metrics.classification_report(
        twenty_test.target, y_pred, output_dict=False, digits=2)

    return results


def experiment2():
    """
    The best classifier with all parameters set to their default values.
    Edit/copy for easy experimentation
    """
    twenty_train = fetch_20newsgroups(
        subset='train', categories=None, shuffle=True, random_state=42)
    twenty_test = fetch_20newsgroups(
        subset='test', categories=None, shuffle=True, random_state=42)

    pipeline = Pipeline([
        ('vect', CountVectorizer(
            lowercase=True,
            stop_words=None,
            max_features=None,
            analyzer='word',
            ngram_range=(1, 1)
        )),
        ('tfidf', TfidfTransformer(use_idf=True)),
        ('clf', SGDClassifier()),
    ])
    data = twenty_train.data
    target = twenty_train.target

    pipeline.fit(data, target)

    y_pred = pipeline.predict(twenty_test.data)
    results = metrics.classification_report(
        # set to true to get as a python dict
        twenty_test.target, y_pred, output_dict=False,
        digits=3)
    print(results)


experiment2()
