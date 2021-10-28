from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount(
    "/static",
    StaticFiles(directory="../frontend/build/static"),
    name="static",
)


@app.get("/", response_class=FileResponse)
async def index():
    return FileResponse("../frontend/build/index.html", media_type="text/html")


@app.get("/hello")
async def hello():
    return {"message": "Hello World"}
