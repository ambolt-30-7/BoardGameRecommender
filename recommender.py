from typing import List
import umap
from topic_model import load_model, get_data

if __name__ == '__main__':
    # Data
    data = get_data("data/database.sqlite", "SELECT * FROM boardgames")

    # Model
    model = load_model('1k_model')

    # Input
    x = "Catan"

    # Parse input
    split_input: List[str] = x.split(' ')

    # Get topics from input
    predictions = model.transform(split_input)

    print(predictions)
