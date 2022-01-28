import itertools
from typing import List

from bertopic import BERTopic
from fastapi import APIRouter

from topic_model.topic_model import get_db_connection


def process_query(query: str):
    # Lowercase query
    query.lower()

    # Parse input
    split_input: List[str] = query.split(' ')

    return split_input


def get_board_game_name_by_desc(desc: str):
    cursor = db.execute("SELECT \"details.name\" FROM boardgames WHERE \"details.description\" LIKE ?", ('%' + desc + '%',))
    return cursor.fetchall()


router = APIRouter()
db = get_db_connection("topic_model/data/database.sqlite")
model = BERTopic.load('topic_model/models/full_model_v2')


@router.get("/item/", tags=["board_game"])
async def get_board_games():
    cursor = db.execute("SELECT * FROM boardgames")
    return cursor.fetchall()


@router.get("/item-by-id", tags=["board_game"])
async def get_board_game_by_id(board_game_id: str):
    cursor = db.execute("SELECT * FROM boardgames WHERE \"game.id\"=?", (board_game_id,))
    return cursor.fetchall()


@router.get("/item-by-description", tags=["board_game"])
async def get_board_game_by_description(description: str):
    cursor = db.execute("SELECT * FROM boardgames WHERE \"details.description\" LIKE ?", ('%' + description + '%',))
    return cursor.fetchall()


@router.get("/item-by-name", tags=["board_game"])
async def get_board_game_by_name(name: str):
    cursor = db.execute("SELECT * FROM boardgames WHERE \"details.name\" LIKE ?", ('%' + name + '%',))
    return cursor.fetchall()


@router.get("/board-game-recommendation", tags=["board_game"])
async def get_board_game_recommendation(query: str):
    # Transform query
    predictions = model.transform(query)

    # Predicted topic
    topics = predictions[0]

    # Representative descriptions from topic
    if topics == -1:
        representative_descriptions = model.get_representative_docs()
    representative_descriptions = model.get_representative_docs(topics)

    # Combine all representative descriptions
    descriptions = list(itertools.chain.from_iterable(list(representative_descriptions.values())))

    return [get_board_game_name_by_desc(x) for x in descriptions]
