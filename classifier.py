import numpy as np

# Step one - read dataset
from pandas import read_csv

url = "https://raw.githubusercontent.com/ash-williams/CS4GL_evaluation/master/data_for_analysis.csv?token=ABHP3RVHEPUMQ3DJZLCFW6264F6W6"
names = [
    "class",
    "rea_total_unique_markers",
    "rea_total_number_of_markers",
    "exp_total_unique_markers",
    "exp_total_number_of_markers",
    "exp_timex_events",
    "exp_verb_events",
    "exp_iverb_events",
    "exp_nltk_named_entities",
    "exp_pronouns",
    "cit_all_uris",
    "cit_external_uris",
    "cit_classified_uris",
    "cit_RESEARCH",
    "cit_RESEARCH_SEARCH",
    "cit_DEVELOPER",
    "cit_EDUCATION",
    "cit_NEWS_AND_MAGAZINES",
    "cit_SOCIAL_MEDIA",
    "cit_GOVERNMENT",
    "cit_Q_AND_A",
    "cit_REPOSITORY",
    "cit_SANDBOX",
    "cit_E_COMMERCE",
    "cit_FORUM",
    "cit_BLOG",
    "cit_JOB_BOARD",
    "cit_WIKI",
    "cit_JOEL",
    "cit_DOCUMENTATION",
    "cit_ASSETS",
    "cit_SHORT_URL",
    "cit_LEARNING",
    "cit_RESOURCES",
    "cit_SUPPORT",
    "cit_ADVERTS",
    "cit_EVENTS",
    "cit_ORGANISATIONS_AND_TECHNOLOGIES",
    "cow_readability_syllable_count",
    "cow_readability_word_count",
    "cow_readability_sentence_count",
    "cow_readability_flesch_reading_ease",
    "cow_readability_flesch_kincaid_grade",
    "cow_readability_gunning_fog",
    "cow_readability_smog_index",
    "cow_readability_automated_readability_index",
    "cow_readability_coleman_liau_index",
    "cow_readability_linsear_write_formula",
    "cow_readability_dale_chall_readability_score",
    "cow_grammar_total_issues",
    "cow_grammar_sentences",
    "cow_grammar_issue_misspelling",
    "cow_grammar_issue_uncategorized",
    "cow_grammar_issue_other",
    "cow_sentiment_polarity",
    "cow_sentiment_subjectivity",
    "cod_total_chars",
    "cod_total_words",
    "cod_total_lines",
    "cod_binary_line_percentage",
    "cod_binary_word_percentage",
    "cod_absolute_word_percentage"
]

dataset = read_csv(url, names=names)

# Step two - is everything working?
print(dataset.shape)
# # print(dataset.head(20))
#
# # print(dataset.describe())
#
# print(dataset.groupby('class').size())

# Step three - plots
from matplotlib import pyplot
#
# # dataset.plot(kind='box', subplots=True, layout=(8,8), sharex=False, sharey=False)
# # pyplot.show()
#
# dataset.hist()
# pyplot.show()

# Step four - validation dataset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

array = dataset.values
X = array[:,1:] #features
y = array[:,0] # classification
print(X)
print(y)

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

np.isnan(np.array(X, dtype=np.float64))
np.isfinite(np.array(X, dtype=np.float64))

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=2, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


# Compare Algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

# Step five = Make predictions
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
