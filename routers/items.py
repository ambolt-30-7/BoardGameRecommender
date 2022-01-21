from fastapi import APIRouter

from topic_model.topic_model import get_db_connection

router = APIRouter()
db = get_db_connection("topic_model/data/database.sqlite")


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
    cursor = db.execute("SELECT * FROM boardgames WHERE \"details.description\" LIKE ?", ('%'+description+'%',))
    return cursor.fetchall()


@router.get("/item-by-name", tags=["board_game"])
async def get_board_game_by_name(name: str):
    cursor = db.execute("SELECT * FROM boardgames WHERE \"details.name\" LIKE ?", ('%'+name+'%',))
    return cursor.fetchall()
