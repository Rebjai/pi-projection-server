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
import eventlet
eventlet.monkey_patch()

import os
import io
import json
import time
import math
import uuid

from typing import Tuple, List, Dict, Any

from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
from dotenv import load_dotenv

# ---- Configuración ----
load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, os.getenv("UPLOAD_FOLDER", "uploads"))
TILES_ROOT = os.path.join(BASE_DIR, os.getenv("TILES_ROOT", "tiles"))
CONFIG_CLIENTS = os.path.join(BASE_DIR, os.getenv("CONFIG_CLIENTS", "configs/clients"))

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
    print(f"[config] loading config for client {client_id}")
    path = client_config_path(client_id)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            print(f"[config] loaded config for client {client_id} from {path}")
            return json.load(f)
    # default config skeleton
    return None

def load_or_create_client_config(client_id: str, default_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = load_client_config(client_id)
    if cfg is None:
        print(f"[config] creating new config for client {client_id} with defaults")
        cfg = default_cfg
        save_client_config(client_id, cfg)
    else:
        print(f"[config] existing config found for client {client_id}")
    return cfg

def save_client_config(client_id: str, cfg: Dict[str, Any]) -> None:
    print(f"[config] saving config for client {client_id}: {cfg}")
    path = client_config_path(client_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    # print(f"[config] saved config for client {client_id} to {path}")

def checkConfigurationUpdates(existing_cfg: Dict[str, Any], new_cfg: Dict[str, Any]) -> bool:
    """
    Check for new fields in new_cfg that are not in existing_cfg.
    If found, add them to existing_cfg and return True.
    Otherwise check for changes in existing fields and update if different.
    Return True if any updates were made, False otherwise.
    """
    print(f"[config] checking for updates in config...")
    print(f"[config] checking for updates in config...")
    print(f"[config] checking for updates in config...")
    updated = False
    for key in new_cfg:
        if key not in existing_cfg and new_cfg[key] is not None:
            existing_cfg[key] = new_cfg[key]
            updated = True
        elif key in existing_cfg and new_cfg[key] is not None and existing_cfg[key] != new_cfg[key]:
            existing_cfg[key] = new_cfg[key]
            updated = True
    print(f"[config] updates found: {updated}")
    return updated

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
    safe_image_basename = image_basename.replace(" ", "_")
    path = os.path.join(TILES_ROOT, safe_image_basename)
    os.makedirs(path, exist_ok=True)
    return path

def tile_output_name(image_basename: str, client_id: str, display_name: str) -> str:
    base = image_basename.replace(" ", "_")
    return f"client_{client_id}_tile_{display_name}.png"

def prepare_tiles_for_image(image_filename: str) -> None:
    """
    Generate tiles for a given image for all clients based on their configs.
    - Rectangles are defined in client canvas coordinates.
    - Rectangles are scaled to the original image resolution.
    - Tiles are saved in: tiles/<image_basename>/client_<client>_tile_<idx>.png
    """
    image_path = os.path.join(UPLOAD_FOLDER, image_filename)
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    # Load original image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    image_height, image_width = image.shape[:2]

    # Process each client configuration
    for cfg_file in os.listdir(CONFIG_CLIENTS):
        client_id = os.path.splitext(cfg_file)[0]
        client_config = load_client_config(client_id)

        process_client_config(
            client_id,
            client_config,
            image,
            image_filename,
            (image_width, image_height),
        )
        # eventlet.spawn_n(
        #     process_client_config,
        #     client_id,
        #     client_config,
        #     image,
        #     image_filename,
        #     (image_width, image_height),
        # )


def process_client_config(
    client_id: str,
    config: Dict[str, Any],
    image,
    image_filename: str,
    image_size: Tuple[int, int],
) -> None:
    """Process a single client configuration and generate image tiles."""
    image_width, image_height = image_size

    canvas_size = config.get("client_canvas_size", {"width": 1920, "height": 1080})
    canvas_width = int(canvas_size.get("width", 1920))
    canvas_height = int(canvas_size.get("height", 1080))

    assignments = config.get("assignments", [])
    if not assignments:
        print(f"[slice] no assignments for client {client_id}, skipping")
        return

    # Ensure tiles directory exists
    image_basename = os.path.splitext(os.path.basename(image_filename))[0]
    tiles_dir = ensure_tile_dir(image_basename)

    for idx, assignment in enumerate(assignments):
        rect = assignment.get("rect", {})
        display_name = assignment.get("display_output", f"display_{idx}")
        tile = extract_tile_from_rect(
            rect, image, (canvas_width, canvas_height), (image_width, image_height)
        )

        if tile is None:
            print(f"[slice] invalid rect for client {client_id} assignment {display_name}, skipping")
            continue

        save_tile(tile, tiles_dir, image_basename, client_id, display_name)
        # eventlet.spawn_n(save_tile, tile, tiles_dir, image_basename, client_id, display_name)


def extract_tile_from_rect(
    rect: Dict[str, Any],
    image,
    canvas_size: Tuple[int, int],
    image_size: Tuple[int, int],
):
    """Extract a tile from the image based on scaled rect coordinates."""
    canvas_width, canvas_height = canvas_size
    image_width, image_height = image_size

    rect_x = float(rect.get("x", 0))
    rect_y = float(rect.get("y", 0))
    rect_w = float(rect.get("w", 0))
    rect_h = float(rect.get("h", 0))

    if rect_w <= 0 or rect_h <= 0:
        return None

    # Scale rect coordinates from canvas → image resolution
    x = int(rect_x / canvas_width * image_width)
    y = int(rect_y / canvas_height * image_height)
    w = int(rect_w / canvas_width * image_width)
    h = int(rect_h / canvas_height * image_height)

    # Clip rect to image boundaries
    x = max(0, min(x, image_width - 1))
    y = max(0, min(y, image_height - 1))
    w = max(1, min(w, image_width - x))
    h = max(1, min(h, image_height - y))

    return image[y:y + h, x:x + w]


def save_tile(tile, tiles_dir: str, image_basename: str, client_id: str, display_name: str) -> None:
    """Save a tile to disk."""
    tile_filename = tile_output_name(image_basename, client_id, display_name)
    tile_path = os.path.join(tiles_dir, tile_filename)
    cv2.imwrite(tile_path, tile)
    print(f"[slice] saved tile for client {client_id}, assignment {display_name} -> {tile_path}")

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
    print(f"[socket] register data: {data}")
    client_id = data.get('client_id')
    meta = data
    if not client_id:
        emit('registered', {'ok': False, 'error': 'missing client_id'})
        return
    connected_clients[client_id] = request.sid
    client_meta[client_id] = meta
    #load config in file and saveConfig in file at CONFIG_CLIENTS updating if needed or creating new
    configdata = meta
    configdata['last_seen'] = int(time.time())
    configdata['is_connected'] = True
    cfg = load_or_create_client_config(client_id, configdata)
        # update existing config with any new fields from meta
    updated = checkConfigurationUpdates(cfg, configdata)
    if updated:
        print(f"[socket] updating config for client {client_id} with new meta fields")
        save_client_config(client_id, configdata)
    print(f"[socket] client registered: {client_id} sid={request.sid} meta={meta}")
    emit('registered', {'ok': True, 'client_id': client_id})

@socketio.on('disconnect')
def handle_disconnect():
    print(f"[socket] disconnect event, sid={request.sid}")
    sid = request.sid
    removed = []
    connected_clients_list = list(connected_clients.items())
    print(f"[socket] currently connected clients: {connected_clients_list}")
    for cid, csid in connected_clients_list:
        if csid == sid:
            if cid in client_meta:
                del client_meta[cid]
                cfg = load_client_config(cid)
                if cfg:
                    cfg['last_seen'] = int(time.time())
                    cfg['is_connected'] = False
                    save_client_config(cid, cfg)
            del connected_clients[cid]
            removed.append(cid)
            print(f"[socket] client disconnected: {cid} sid={sid}")
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
    print(f"[uploads] {len(files)} files")
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
    print(connected_clients)
    connected = list(connected_clients.keys())
    connectedClientConfigs = []
    for cid in connected:
        cfg = load_client_config(cid)
        connectedClientConfigs.append({'client_id': cid, 'config': cfg})

    return jsonify({'connected': connectedClientConfigs})

@app.route("/config/<client_id>", methods=["GET", "POST"])
def config_client(client_id):
    """
    GET: return client config JSON
    POST: update client config JSON (partial update)
    Expected config fields:
    - client_id: str
    - displays: list of { "name":str, "resolution": {width:int, height:int}, "status": "connected"/"disconnected", "active": bool}
    - assignments: list of { "display_name":str, rect: {x,y,width,height}
    - grid_resolution: {width:int, height:int}
    - scale_mode: "fill" or "fit"
    """
    if request.method == "GET":
        cfg = load_client_config(client_id)
        return jsonify(cfg)
    else:
        data = request.get_json()
        if not data:
            return jsonify({'ok': False, 'error': 'no json body'}), 400
        # basic validation: ensure grid_resolution is two ints etc
        try:
            if 'client_id' in data and data['client_id'] != client_id:
                raise ValueError("client_id in body must match URL")
            if 'displays' in data:
                displays = data['displays']
                if not (isinstance(displays, list) and all(isinstance(d, dict) and 'name' in d and 'resolution' in d and isinstance(d['resolution'], dict) and 'width' in d['resolution'] and 'height' in d['resolution'] for d in displays)):
                    raise ValueError("displays must be list of {name:str, resolution:{width:int,height:int}, status:str, active:bool}")
            if 'client_canvas_size' in data:
                gr = data['client_canvas_size']
                if not (isinstance(gr, dict) and 'width' in gr and 'height' in gr and isinstance(gr['width'], int) and isinstance(gr['height'], int)):
                    raise ValueError("client_canvas_size must be {width:int, height:int}")
            if 'scale_mode' in data:
                sm = data['scale_mode']
                if sm not in ('fill', 'fit'):
                    raise ValueError("scale_mode must be 'fill' or 'fit'")
            if 'assignments' in data:
                assignments = data['assignments']
                if not (isinstance(assignments, list) and all(isinstance(a, dict) and 'display_output' in a and 'rect' in a for a in assignments)):
                    raise ValueError("assignments must be list of {display_name:str, rect:{x:int,y:int,w:int,h:int}}")
            

            cfg = load_or_create_client_config(client_id, data)
            updated = checkConfigurationUpdates(cfg, data)
            print(f"[config] config update for client {client_id}, updated={updated}")
            if updated:
                save_client_config(client_id, cfg)
                # if client connected, emit CONFIG event with new config
                sid = connected_clients.get(client_id)
                if sid:
                    print(f"[config] emitting CONFIG event to client {client_id}")
                    socketio.emit('CONFIG', cfg, room=sid)
                return jsonify({'ok': True, 'updated': True, 'config': cfg})
            else:
                print(f"[config] no changes in config for client {client_id}")
                return jsonify({'ok': True, 'updated': False, 'config': cfg})
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)}), 400

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
            # eventlet.spawn_n(prepare_tiles_for_image, f)
            print(f"[job {job_id}] sliced {f}")
            #send event to client to get slice
            socketio.emit("FETCH_TILES", {"filename": f})
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
            fname = tile_output_name(image_basename, client_id, display_name)
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
    SERVER_HOST = os.getenv("SERVER_HOST", "127.0.0.1")
    SERVER_PORT = int(os.getenv("SERVER_PORT", "5000"))
    print("Starting server on http://{}:{}".format(SERVER_HOST, SERVER_PORT))
    # socketio.run(app, host=SERVER_HOST, port=SERVER_PORT)
    eventlet.wsgi.server(eventlet.listen((SERVER_HOST, SERVER_PORT)), app)
    print("Server stopped")
