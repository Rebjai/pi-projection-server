# server.py
"""
Flask + Flask-SocketIO server for tiling/distribution + client config management.

Funciones:
- /upload                 : subir imagen al folder uploads
- /uploads                : listar uploads
- /config/<client_id>     : GET/POST config JSON por cliente
- /config/<client_id>/homography/<tile_idx> : POST para guardar/computar H
- /slice_image            : POST filename -> genera tiles para esa imagen
- /slice_all              : POST -> lanza slicing para todas las imágenes en uploads
- /distribute             : POST {"image": "name.png"} -> emitir ASSIGN_TILES a clients conectados
- /tiles/<image>/<file>   : servir archivos de tiles (PNG)
- /clients                : listar clientes conectados + configs
- SocketIO 'register'     : clientes se registran (client_id, capabilities)
- SocketIO 'ASSIGN_TILES' : enviado por server, cliente GETea tiles y los muestra
- SocketIO 'SHOW'         : orden de mostrar sincronizado (frame_id)
"""

import os
import io
import json
import time
import math
import uuid
import eventlet
eventlet.monkey_patch()

from typing import Tuple, List, Dict, Any

from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import cv2
import numpy as np

# ---- Configuración ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
TILES_ROOT = os.path.join(BASE_DIR, "tiles")
CONFIG_CLIENTS = os.path.join(BASE_DIR, "configs", "clients")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TILES_ROOT, exist_ok=True)
os.makedirs(CONFIG_CLIENTS, exist_ok=True)

# Flask + SocketIO
app = Flask(__name__)
CORS(app)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
socketio = SocketIO(app, cors_allowed_origins="*")

# Map client_id -> socket sid
connected_clients: Dict[str, str] = {}
# Optional client meta data (capabilities) captured at register time
client_meta: Dict[str, Dict[str, Any]] = {}

# ---- Helpers: config load/save ----
def client_config_path(client_id: str) -> str:
    safe = str(client_id).replace("/", "_")
    return os.path.join(CONFIG_CLIENTS, f"{safe}.json")

def load_client_config(client_id: str) -> Dict[str, Any]:
    path = client_config_path(client_id)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    # default config skeleton
    return {
        "client_id": client_id,
        "grid_resolution": [1920, 1080],
        "tile_indexes": [],            # ex: [0,1]
        "hdmi_outputs": [],            # ex: [0,1]
        "tile_coordinates": [],        # ex: [ [[x1,y1],[x2,y2]], ... ] per tile
        "scale_mode": "fill",          # "fill" or "fit"
        "homographies": {}             # "tile_index": [[..],...]
    }

def save_client_config(client_id: str, cfg: Dict[str, Any]) -> None:
    path = client_config_path(client_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

# ---- Image scaling / slicing utilities ----
def scale_image_to_grid(img: np.ndarray, grid_w: int, grid_h: int, mode: str = "fill") -> np.ndarray:
    """
    Scale an image to target grid size.
    mode "fill": scale to cover grid then center-crop
    mode "fit" : scale to fit within grid, pad with black
    """
    h, w = img.shape[:2]
    if w == 0 or h == 0:
        raise ValueError("Invalid image size")
    if mode == "fill":
        scale = max(grid_w / w, grid_h / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # center crop
        x0 = (new_w - grid_w) // 2
        y0 = (new_h - grid_h) // 2
        cropped = resized[y0:y0 + grid_h, x0:x0 + grid_w]
        return cropped
    else:  # fit
        scale = min(grid_w / w, grid_h / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((grid_h, grid_w, 3), dtype=resized.dtype)
        x0 = (grid_w - new_w) // 2
        y0 = (grid_h - new_h) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
        return canvas

def ensure_tile_dir(image_basename: str) -> str:
    safe = image_basename.replace(" ", "_")
    path = os.path.join(TILES_ROOT, safe)
    os.makedirs(path, exist_ok=True)
    return path

def tile_output_name(image_basename: str, client_id: str, tile_index: int) -> str:
    base = image_basename.replace(" ", "_")
    return f"client_{client_id}_tile_{tile_index}.png"

def prepare_tiles_for_image(image_filename: str):
    """
    Generate tiles for a given image for all clients based on their configs.
    Save tiles in tiles/<image_basename>/client_<client>_tile_<idx>.png
    """
    image_path = os.path.join(UPLOAD_FOLDER, image_filename)
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    # Load image with OpenCV (BGR)
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Cannot read image: " + image_path)

    # For each client config, create tile images based on that client's grid_resolution and tile coordinates
    for cfg_file in os.listdir(CONFIG_CLIENTS):
        client_id = os.path.splitext(cfg_file)[0]
        cfg = load_client_config(client_id)
        grid_res = cfg.get("grid_resolution", [1920, 1080])
        grid_w, grid_h = int(grid_res[0]), int(grid_res[1])
        scale_mode = cfg.get("scale_mode", "fill")
        tile_coords = cfg.get("tile_coordinates", [])
        tile_indexes = cfg.get("tile_indexes", [])
        if not tile_coords or not tile_indexes:
            # nothing to produce for this client
            continue
        # scale image to grid
        try:
            scaled = scale_image_to_grid(img_bgr, grid_w, grid_h, scale_mode)
        except Exception as e:
            print(f"[tiler] error scaling for client {client_id}: {e}")
            continue
        # ensure output dir
        image_basename = os.path.splitext(os.path.basename(image_filename))[0]
        out_dir = ensure_tile_dir(image_basename)
        # iterate assigned tiles (zip tile_indexes + coords)
        if len(tile_indexes) != len(tile_coords):
            print(f"[tiler] WARNING: client {client_id} tile_indexes length != tile_coords length. Skipping mismatched.")
        pairs = list(zip(tile_indexes, tile_coords))
        for (tidx, coord) in pairs:
            try:
                # coord expected [[x1,y1],[x2,y2]]
                (x1y1, x2y2) = coord
                x1, y1 = int(round(x1y1[0])), int(round(x1y1[1]))
                x2, y2 = int(round(x2y2[0])), int(round(x2y2[1]))
                # clip to grid
                x1 = max(0, min(x1, grid_w - 1))
                x2 = max(0, min(x2, grid_w))
                y1 = max(0, min(y1, grid_h - 1))
                y2 = max(0, min(y2, grid_h))
                if x2 <= x1 or y2 <= y1:
                    print(f"[tiler] invalid coords for client {client_id} tile {tidx}: {coord}")
                    continue
                roi = scaled[y1:y2, x1:x2]  # Note: y first in numpy
                out_name = tile_output_name(image_basename, client_id, tidx)
                out_path = os.path.join(out_dir, out_name)
                cv2.imwrite(out_path, roi)
                # optionally save small metadata maybe later
            except Exception as e:
                print(f"[tiler] error slicing tile {tidx} for client {client_id}: {e}")

# ---- Homography utilities ----
def compute_h_from_points(src_pts: List[List[float]], dst_pts: List[List[float]]) -> List[List[float]]:
    """
    src_pts and dst_pts expected as lists of 4 [x,y] points
    returns 3x3 matrix as list of lists (float)
    """
    if len(src_pts) != 4 or len(dst_pts) != 4:
        raise ValueError("src_pts and dst_pts must be lists of 4 points")
    src = np.array(src_pts, dtype=np.float32)
    dst = np.array(dst_pts, dtype=np.float32)
    H, status = cv2.findHomography(src, dst, method=cv2.RANSAC)
    if H is None:
        raise RuntimeError("findHomography failed")
    return H.tolist()

# ---- SocketIO handlers ----
@socketio.on('register')
def handle_register(data):
    client_id = data.get('client_id')
    meta = data.get('meta', {})
    if not client_id:
        emit('registered', {'ok': False, 'error': 'missing client_id'})
        return
    connected_clients[client_id] = request.sid
    client_meta[client_id] = meta
    print(f"[socket] client registered: {client_id} sid={request.sid} meta={meta}")
    emit('registered', {'ok': True, 'client_id': client_id})

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    removed = []
    for cid, csid in list(connected_clients.items()):
        if csid == sid:
            del connected_clients[cid]
            client_meta.pop(cid, None)
            removed.append(cid)
    if removed:
        print(f"[socket] disconnected clients removed: {removed}")

# ---- HTTP endpoints ----

@app.route("/upload", methods=["POST"])
def upload_file():
    """
    multipart/form-data: file=@file.png
    """
    if 'file' not in request.files:
        return jsonify({'ok': False, 'error': 'no file field'}), 400
    f = request.files['file']
    filename = f.filename
    if filename == '':
        return jsonify({'ok': False, 'error': 'empty filename'}), 400
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    f.save(save_path)
    return jsonify({'ok': True, 'filename': filename})

@app.route("/uploads", methods=["GET"])
def list_uploads():
    files = [f for f in os.listdir(UPLOAD_FOLDER) if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))]
    return jsonify({'uploads': files})

@app.route("/uploads/<filename>", methods=["GET"])
def serve_upload(filename):
    """
    Serve uploaded files
    """
    safe_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(safe_path):
        return abort(404)
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/tiles/<image_basename>/<tile_file>", methods=["GET"])
def serve_tile(image_basename, tile_file):
    """
    Serve generated tile files: tiles/<image_basename>/
    """
    safe_dir = os.path.join(TILES_ROOT, image_basename)
    if not os.path.exists(os.path.join(safe_dir, tile_file)):
        return abort(404)
    return send_from_directory(safe_dir, tile_file)

@app.route("/clients", methods=["GET"])
def list_clients():
    """
    Return connected clients + known client configs on disk
    """
    connected = list(connected_clients.keys())
    configs = []
    for cfg in os.listdir(CONFIG_CLIENTS):
        cid = os.path.splitext(cfg)[0]
        configs.append(cid)
    return jsonify({'connected': connected, 'known_configs': configs})

@app.route("/config/<client_id>", methods=["GET", "POST"])
def config_client(client_id):
    if request.method == "GET":
        cfg = load_client_config(client_id)
        return jsonify(cfg)
    else:
        data = request.get_json()
        if not data:
            return jsonify({'ok': False, 'error': 'no json body'}), 400
        # basic validation: ensure grid_resolution is two ints etc
        try:
            # coerce certain fields
            if 'grid_resolution' in data:
                gr = data['grid_resolution']
                data['grid_resolution'] = [int(gr[0]), int(gr[1])]
            if 'tile_indexes' in data:
                data['tile_indexes'] = [int(i) for i in data['tile_indexes']]
            if 'hdmi_outputs' in data:
                data['hdmi_outputs'] = [int(i) for i in data['hdmi_outputs']]
            # tile_coords should be nested lists - keep as is
        except Exception as e:
            return jsonify({'ok': False, 'error': f'validation error: {e}'}), 400
        save_client_config(client_id, data)
        return jsonify({'ok': True})

@app.route("/config/<client_id>/homography/<int:tile_idx>", methods=["POST"])
def set_homography(client_id, tile_idx):
    """
    POST body:
    - either {"H": [[...],[...],[...]]}  (3x3)
    - or {"src": [[x,y],...4], "dst": [[x,y],...4]} -> server computes H
    """
    data = request.get_json()
    if not data:
        return jsonify({'ok': False, 'error': 'no json body'}), 400
    cfg = load_client_config(client_id)
    homos = cfg.get('homographies', {})
    try:
        if 'H' in data:
            H = data['H']
            Hn = np.array(H, dtype=np.float32)
            if Hn.shape != (3,3):
                raise ValueError("H must be 3x3")
            homos[str(tile_idx)] = H
        elif 'src' in data and 'dst' in data:
            H = compute_h_from_points(data['src'], data['dst'])
            homos[str(tile_idx)] = H
        else:
            return jsonify({'ok': False, 'error': 'provide H matrix or src/dst points'}), 400
        cfg['homographies'] = homos
        save_client_config(client_id, cfg)
        return jsonify({'ok': True, 'H': homos[str(tile_idx)]})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route("/slice_image", methods=["POST"])
def slice_image():
    """
    POST JSON: {"image": "name.png"}
    returns: job_id
    """
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'ok': False, 'error': 'image param required'}), 400
    image = data['image']
    image_path = os.path.join(UPLOAD_FOLDER, image)
    if not os.path.exists(image_path):
        return jsonify({'ok': False, 'error': 'image not found'}), 404

    job_id = str(uuid.uuid4())
    # spawn background task to avoid blocking
    eventlet.spawn_n(_slice_image_job, image, job_id)
    return jsonify({'ok': True, 'job_id': job_id})

def _slice_image_job(image, job_id):
    print(f"[job {job_id}] starting slicing for {image}")
    try:
        prepare_tiles_for_image(image)
        print(f"[job {job_id}] finished slicing {image}")
    except Exception as e:
        print(f"[job {job_id}] ERROR slicing {image}: {e}")

@app.route("/slice_all", methods=["POST"])
def slice_all():
    """
    Launch slicing for all images in uploads.
    """
    files = [f for f in os.listdir(UPLOAD_FOLDER) if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))]
    job_id = str(uuid.uuid4())
    eventlet.spawn_n(_slice_all_job, files, job_id)
    return jsonify({'ok': True, 'job_id': job_id, 'files_count': len(files)})

def _slice_all_job(files, job_id):
    print(f"[job {job_id}] slice_all start: {len(files)} files")
    for f in files:
        try:
            prepare_tiles_for_image(f)
            print(f"[job {job_id}] sliced {f}")
        except Exception as e:
            print(f"[job {job_id}] error slicing {f}: {e}")
    print(f"[job {job_id}] slice_all finished")

@app.route("/distribute", methods=["POST"])
def distribute():
    """
    POST JSON: {"image": "name.png"}  -> emit ASSIGN_TILES to connected clients
    If image is not provided, use last uploaded or return error.
    """
    data = request.get_json() or {}
    image = data.get('image')
    if not image:
        # choose latest upload
        uploads = sorted([f for f in os.listdir(UPLOAD_FOLDER) if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))],
                         key=lambda x: os.path.getmtime(os.path.join(UPLOAD_FOLDER, x)), reverse=True)
        if not uploads:
            return jsonify({'ok': False, 'error': 'no uploads available'}), 400
        image = uploads[0]
    image_basename = os.path.splitext(os.path.basename(image))[0]
    tiles_dir = os.path.join(TILES_ROOT, image_basename)
    if not os.path.exists(tiles_dir):
        return jsonify({'ok': False, 'error': 'tiles not generated for this image (run slice_image first)'}), 400

    # build and send per-client payloads
    host = request.host_url.rstrip('/')
    sent = []
    failed = []
    for cfg_file in os.listdir(CONFIG_CLIENTS):
        client_id = os.path.splitext(cfg_file)[0]
        cfg = load_client_config(client_id)
        tile_indexes = cfg.get('tile_indexes', [])
        hdmi_outputs = cfg.get('hdmi_outputs', [])
        if len(tile_indexes) != len(hdmi_outputs):
            # mismatch -> continue but warn
            print(f"[distribute] WARNING mismatch tile_indexes vs hdmi_outputs for {client_id}")
        tiles_payload = []
        for idx, tidx in enumerate(tile_indexes):
            # create filename and URL per convention
            fname = tile_output_name(image_basename, client_id, tidx)
            local_path = os.path.join(tiles_dir, fname)
            if not os.path.exists(local_path):
                print(f"[distribute] missing tile {local_path} for client {client_id}, tile {tidx}")
                continue
            url = f"{host}/tiles/{image_basename}/{fname}"
            hdmi = hdmi_outputs[idx] if idx < len(hdmi_outputs) else 0
            homos = cfg.get('homographies', {})
            H = homos.get(str(tidx)) if homos else None
            tiles_payload.append({'tile_index': tidx, 'url': url, 'hdmi_output': hdmi, 'homography': H})
        # emit only if client connected
        sid = connected_clients.get(client_id)
        if sid:
            payload = {'image': image, 'tiles': tiles_payload, 'frame_id': int(time.time())}
            socketio.emit('ASSIGN_TILES', payload, room=sid)
            sent.append(client_id)
            print(f"[distribute] ASSIGN_TILES -> {client_id} (tiles: {len(tiles_payload)})")
        else:
            failed.append(client_id)
            print(f"[distribute] client {client_id} not connected")
    return jsonify({'ok': True, 'sent': sent, 'failed': failed})

@app.route("/show", methods=["POST"])
def show():
    """
    POST param frame_id optional: broadcast SHOW to all connected clients
    """
    frame_id = int(request.args.get('frame_id', time.time()))
    for cid, sid in connected_clients.items():
        socketio.emit('SHOW', {'frame_id': frame_id}, room=sid)
    return jsonify({'ok': True, 'frame_id': frame_id, 'clients': list(connected_clients.keys())})

# ---- Main ----
if __name__ == "__main__":
    print("Starting server on 0.0.0.0:5000")
    socketio.run(app, host="0.0.0.0", port=5000)
