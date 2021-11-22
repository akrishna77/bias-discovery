from flask import Flask

app = Flask(__name__)
app.secret_key = "GT903476634"

from app import views
