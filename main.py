import sqlite3
import time
from typing import List

from nltk.corpus import stopwords
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore, CoherenceModel


def get_data(path: str, query: str):
    connection = sqlite3.connect(path)
    cursor = connection.cursor().execute(query)
    full_data = cursor.fetchall()
    return full_data


def preprocessing(texts: List[str]):
    cleaned_texts = [remove_stopwords(s.lower(), stopwords.words('english')) for s in texts]
    tokenized_texts = [x.split(' ') for x in cleaned_texts]
    dictionary = Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    return tokenized_texts, dictionary, corpus


def make_lda(texts: List[str], num_topics: int, model_name: str):
    texts, dictionary, corpus = preprocessing(texts)
    model = LdaMulticore(corpus, id2word=dictionary, num_topics=num_topics)
    evaluate_coherence_and_perplexity(model, texts, dictionary, corpus)
    save("models/", model, model_name)
    return model


def save(path: str, lda_model: LdaMulticore, model_name: str):
    lda_model.save(path + model_name + str(lda_model.num_topics))


def load(path: str):
    return LdaMulticore.load(path)


def evaluate_coherence_and_perplexity(lda_model, texts, dictionary, corpus):
    # Compute Perplexity
    # a measure of how good the model is. lower the better.
    print('\nPerplexity: ', lda_model.log_perplexity(corpus))
    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(
        model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)


def compute_coherence_values(dictionary, corpus, texts, test_range):
    """
    Compute c_v coherence for various number of topics
    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics
    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in test_range:
        model = LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        model_list.append(model)
        coherencemodel = CoherenceModel(
            model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values


def search_num_topics(dictionary, corpus, texts, test_range):
    model_list, coherence_values = compute_coherence_values(dictionary=dictionary,
                                                            corpus=corpus,
                                                            texts=texts,
                                                            test_range=test_range)
    # Print the coherence scores
    numtopics_coherence = zip(test_range, coherence_values)
    for m, cv in numtopics_coherence:
        print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
    # pasta friendly
    print(coherence_values)
    max_index = coherence_values.index(max(coherence_values))
    return model_list[max_index]


if __name__ == '__main__':
    descriptions = [x[3] for x in get_data("data/database.sqlite", "SELECT * FROM boardgames")][:1000]
    # lda = make_lda(descriptions, 10, "lda_model_lower")
    # lda = load('models/lda_model')
    # print("Hello")

    from bertopic import BERTopic

    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(descriptions)
    topic_model.save("My_model")
    topic_model.visualize_topics()

