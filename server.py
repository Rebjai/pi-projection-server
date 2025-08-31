# server.py
from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import os, cv2, numpy as np

UPLOAD_FOLDER = "uploads"
TILES_FOLDER = "tiles"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
socketio = SocketIO(app, cors_allowed_origins="*")

clients = {}

@app.route("/upload", methods=["POST"])
def upload_file():
    f = request.files["file"]
    path = os.path.join(UPLOAD_FOLDER, f.filename)
    f.save(path)
    return jsonify({"status": "ok", "filename": f.filename})

@socketio.on("register")
def register(data):
    cid = data["client_id"]
    clients[cid] = request.sid
    emit("registered", {"msg": f"Hello client {cid}"})

def send_tile(client_id, tile_path):
    with open(tile_path, "rb") as f:
        img_bytes = f.read()
    socketio.emit("tile", {"filename": os.path.basename(tile_path), "data": img_bytes.hex()}, room=clients[client_id])

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(TILES_FOLDER, exist_ok=True)
    socketio.run(app, host="0.0.0.0", port=5000)
