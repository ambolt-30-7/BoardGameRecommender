from typing import List

from topic_model import load_model, get_db_connection


def process_query(query: str):
    # Lowercase query
    query.lower()

    # Parse input
    split_input: List[str] = query.split(' ')

    return split_input


if __name__ == '__main__':
    # Data
    data = get_db_connection("data/database.sqlite", "SELECT * FROM boardgames")

    # Model
    model = load_model('full_model')

    # Input
    x = "Deck-Building"

    # Get topics from input
    predictions = model.transform(process_query(x))

    # Predicted topic
    topic = int(predictions[0][0])

    # Representative descriptions from topic
    representative_descriptions = model.get_representative_docs(topic)

    for description in representative_descriptions:
        print(description)
