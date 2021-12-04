import sqlite3
from typing import List

from bertopic import BERTopic


def get_data(path: str, query: str):
    connection = sqlite3.connect(path)
    # connection.row_factory = sqlite3.Row
    cursor = connection.cursor().execute(query)
    full_data = cursor.fetchall()
    return full_data


def query_data(path: str, query):
    connection = sqlite3.connect(path)
    result_set = connection.execute(query)
    return result_set.fetchall()


def train_model(data: List[str], name: str) -> List[int]:
    """
    Trains a model and saves it as name in the folder models
    :param name: the wanted name for the model
    :param data: list of strings
    :return: the topics
    """
    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(data)
    topic_model.save("models/" + name)
    return topics


def load_model(name: str) -> BERTopic:
    return BERTopic.load('models/' + name)


if __name__ == '__main__':
    descriptions = [x[3] for x in get_data("data/database.sqlite", "SELECT * FROM boardgames")]
    train_model(descriptions, 'full_model')
