import uvicorn
from fastapi import FastAPI

from routers import items

app = FastAPI()
app.include_router(router=items.router)


@app.get("/")
def read_root():
    return {"Hello": "World"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
