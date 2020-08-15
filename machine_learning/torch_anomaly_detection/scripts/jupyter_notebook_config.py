import os

c.NotebookApp.ip = "127.0.0.1"
c.NotebookApp.open_browser = False
c.NotebookApp.port = int(os.environ.get("JUPYTER_PORT", "8080"))
c.NotebookApp.token = "jupyter"
