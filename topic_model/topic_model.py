import sqlite3
from sqlite3 import Connection
from typing import List

import pandas
import pandasql as ps
from bertopic import BERTopic


def get_db_connection(path: str) -> Connection:
    return sqlite3.connect(path)


def get_full_data_and_column_names(path: str, query: str):
    connection = get_db_connection(path)
    cursor = connection.cursor().execute(query)
    column_names = list(map(lambda x: x[0], cursor.description))
    full_data = cursor.fetchall()
    return column_names, full_data


def get_data(path: str) -> pandas.DataFrame:
    conn = sqlite3.connect(path)
    df = pandas.read_sql_query("SELECT * FROM BoardGames", conn)
    conn.close()
    return df


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
    # columns, board_games = get_full_data_and_column_names("data/database.sqlite", "SELECT * FROM boardgames")
    # descriptions = [x[3] for x in board_games]
    connection = get_db_connection("data/database.sqlite")
    train_model(descriptions, 'full_model')
