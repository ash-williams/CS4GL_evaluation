
# Step one - read dataset
from pandas import read_csv

url = "https://raw.githubusercontent.com/ash-williams/CS4GL_evaluation/master/data_for_analysis.csv?token=ABHP3RSKY7XUUDZLPLRGZHC64F2RQ"
names = [
"ID", "Jim - Annotation", "rea_total_unique_markers", "rea_total_number_of_markers", "exp_total_unique_markers", "exp_total_number_of_markers", "exp_timex_events", "exp_verb_events", "exp_iverb_events", "exp_nltk_named_entities", "exp_pronouns", "cit_all_uris", "cit_external_uris", "cit_classified_uris", "cit_RESEARCH", "cit_RESEARCH_SEARCH", "cit_DEVELOPER", "cit_EDUCATION", "cit_NEWS_AND_MAGAZINES", "cit_SOCIAL_MEDIA", "cit_GOVERNMENT", "cit_Q_AND_A", "cit_REPOSITORY", "cit_SANDBOX", "cit_E_COMMERCE", "cit_FORUM", "cit_BLOG", "cit_JOB_BOARD", "cit_WIKI", "cit_JOEL", "cit_DOCUMENTATION", "cit_ASSETS", "cit_SHORT_URL", "cit_LEARNING", "cit_RESOURCES", "cit_SUPPORT", "cit_ADVERTS", "cit_EVENTS", "cit_ORGANISATIONS_AND_TECHNOLOGIES", "cow_readability_syllable_count", "cow_readability_word_count", "cow_readability_sentence_count", "cow_readability_flesch_reading_ease", "cow_readability_flesch_kincaid_grade", "cow_readability_gunning_fog", "cow_readability_smog_index,cow_readability_automated_readability_index", "cow_readability_coleman_liau_index", "cow_readability_linsear_write_formula", "cow_readability_dale_chall_readability_score", "cow_grammar_total_issues", "cow_grammar_sentences", "cow_grammar_issue_misspelling", "cow_grammar_issue_uncategorized", "cow_grammar_issue_other", "cow_sentiment_polarity", "cow_sentiment_subjectivity", "cod_total_chars", "cod_total_words", "cod_total_lines", "cod_binary_line_percentage", "cod_binary_word_percentage", "cod_absolute_word_percentagecod_binary_line_percentage", "cod_binary_word_percentage", "cod_absolute_word_percentagecod_binary_line_percentage", "cod_binary_word_percentage", "cod_absolute_word_percentage"
]

dataset = read_csv(url, names=names)

# Step two - is everything working?
print(dataset.shape)
print(dataset.head(20))
