import sqlite3

from bertopic import BERTopic


def get_data(path: str, query: str):
    connection = sqlite3.connect(path)
    cursor = connection.cursor().execute(query)
    full_data = cursor.fetchall()
    return full_data


if __name__ == '__main__':
    descriptions = [x[3] for x in get_data("data/database.sqlite", "SELECT * FROM boardgames")][:1000]

    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(descriptions)
    topic_model.save("models/My_model")
    topic_model.visualize_barchart()
