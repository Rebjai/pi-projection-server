# server.py
"""
Flask + Flask-SocketIO server for tiling/distribution + client config management.

Funciones:
- /upload                 : subir imagen al folder uploads
- /uploads                : listar uploads
- /config/<client_id>     : GET/POST config JSON por cliente
- /config/<client_id>/homography/<tile_idx> : POST para guardar/computar H
- /slice_image            : POST filename -> genera tiles para esa imagen
- /slice_all              : POST -> emite FETCH_TILES a clients conectados para que bajen las uploads y las sliceen
- /start_presentation     : POST -> iniciar presentación (emite START_PRESENTATION a clients conectados)
- /stop_presentation      : POST -> detener presentación
- /manual_show/<image>    : POST -> mostrar imagen y pausar presentación
- /presentation/next      : POST -> avanzar a siguiente imagen y pausar presentación
- /presentation/prev      : POST -> retroceder a imagen anterior y pausar presentación
- /calibrate/<client_id>/<display_name> : POST -> poner cliente en modo calibración para display
- /calibrate/<client_id>/<display_name>/exit : POST -> salir de modo calibración
- /tiles/<image>/<file>   : servir archivos de tiles (PNG)
- /clients                : listar clientes conectados + configs
- SocketIO 'register'     : clientes se registran (client_id, capabilities)
- SocketIO 'ASSIGN_TILES' : enviado por server, cliente GETea tiles y los muestra
- SocketIO 'SHOW'         : orden de mostrar sincronizado (frame_id)
- SocketIO 'PRESENTATION_READY' : enviado por cliente cuando está listo para presentación
- SocketIO 'IMAGE_SHOWN'  : ack de cliente cuando imagen fue mostrada
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

slideshow_running = False
slideshow_paused = False
slideshow_job = None
current_index = 0
resume_timer = None

PRESENTATION_INTERVAL = 10  # seconds
RESUME_TIMEOUT = 30        # seconds

images_list = []  # set this when slicing is done
clients_ready = set()  # clients that sent READY
expected_clients = set()  # clients expected to join

# State for the current slice_all run
slice_event = None
waiting_for = set()

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

def handleClientConfigUpdate(client_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
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

# ---- presentation / slideshow utilities ----
def start_presentation(files):
    """start presentation variables and notify clients to wait for READY"""
    global slideshow_running, images_list, current_index, slideshow_job, slideshow_paused, resume_timer
    if slideshow_running:
        print("[presentation] already running, ignoring start request")
        return
    if not files:
        print("[presentation] no files provided, cannot start")
        return
    images_list = files
    current_index = 0
    slideshow_running = True
    slideshow_paused = False

    # notify clients
    clients_ready.clear()
    expected_clients.clear()
    for cid in connected_clients.keys():
        expected_clients.add(cid)
    print(f"[presentation] expecting {len(expected_clients)} clients to join: {expected_clients}")
    socketio.emit("START_PRESENTATION", {"images": files})
    print(f"[presentation] started with {len(files)} images")
    return
    

def stop_presentation():
    """Stop presentation loop"""
    global slideshow_running, slideshow_paused, slideshow_job, resume_timer
    slideshow_running = False
    slideshow_paused = False
    if slideshow_job:
        slideshow_job.kill()
        slideshow_job = None
    if resume_timer:
        resume_timer.cancel()
        resume_timer = None
    return

def slideshow_loop():
    """Loop that emits SHOW_IMAGE events"""
    global current_index, slideshow_running, slideshow_paused

    while slideshow_running:
        if not slideshow_paused and images_list:
            image = images_list[current_index % len(images_list)]
            print(f"[slideshow] showing {image}")
            socketio.emit("SHOW_IMAGE", {"image": image})
            current_index += 1

        # wait interval
        eventlet.sleep(PRESENTATION_INTERVAL)
        # if paused, just wait without changing image
        while slideshow_paused and slideshow_running:
            eventlet.sleep(1)
    print(f"[slideshow] loop exited")
    return

def manual_show(image):
    """Manual override: show image and pause slideshow"""
    global slideshow_paused, resume_timer, slideshow_running

    slideshow_paused = True
    socketio.emit("SHOW_IMAGE", {"image": image})
    print(f"[manual] override: {image}")

    # cancel any existing resume timer
    if slideshow_running and resume_timer:
        resume_timer.cancel()

    # schedule resume
    if slideshow_running:
        resume_timer = eventlet.spawn_after(RESUME_TIMEOUT, resume_presentation)


def resume_presentation():
    """Resume slideshow after manual override"""
    global slideshow_paused
    slideshow_paused = False
    socketio.emit("RESUME_PRESENTATION", {})
    print("[slideshow] resumed")
    return

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
        return handleClientConfigUpdate(client_id, data)
    
@app.route("/configs", methods=["GET", "POST"])
def list_all_configs():
    """
    GET: list all client configs on disk
    POST: bulk update multiple client configs
    Body: {"configs": [ {client_id: str, ...}, ... ] }
    """
    if request.method == "GET":
        configs = []
        for cfg_file in os.listdir(CONFIG_CLIENTS):
            client_id = os.path.splitext(cfg_file)[0]
            cfg = load_client_config(client_id)
            if cfg:
                configs.append({'client_id': client_id, 'config': cfg})
        return jsonify({'configs': configs})
    else:
        data = request.get_json()
        if not data or 'configs' not in data:
            return jsonify({'ok': False, 'error': 'no json body or missing configs field'}), 400
        results = []
        for cfg in data['configs']:
            client_id = cfg.get('client_id')
            if not client_id:
                results.append({'ok': False, 'error': 'missing client_id in one of the configs'})
                continue
            print(f"[bulk config] updating config for client {client_id}")
            res = handleClientConfigUpdate(client_id, cfg)
            results.append(res.get_json())
        return jsonify({'results': results})


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
    global slice_event, waiting_for

    files = [
        f for f in os.listdir(UPLOAD_FOLDER)
        if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))
    ]

    # Create a new event + snapshot clients
    slice_event = eventlet.event.Event()
    waiting_for = set(connected_clients.keys())

    print(f"[slice_all] expecting {len(waiting_for)} clients: {waiting_for}")

    # Tell all clients to start
    socketio.emit("FETCH_ALL_IMAGES", {"images": files})

    try:
        # Wait until all clients respond (with timeout)
        slice_event.wait(timeout=60)
    except eventlet.timeout.Timeout:
        missing = list(waiting_for)
        return jsonify({"ok": False, "error": "timeout", "waiting_for": missing}), 504

    return jsonify({"ok": True, "files_count": len(files)})

@socketio.on("ALL_TILES_READY")
def handle_all_tiles_ready(data):
    client_id = data.get("client_id")
    global slice_event, waiting_for

    if slice_event and client_id in waiting_for:
        waiting_for.discard(client_id)
        print(f"[socket] {client_id} is ready, still waiting for: {waiting_for}")

        if not waiting_for:
            slice_event.send("done")  # resolve the event
            slice_event = None
            print("[socket] all clients sliced tiles")
    return 

# ---- Presentation control ----
@app.route("/start_presentation", methods=["POST"])
def api_start_presentation():
    files = [f for f in os.listdir(UPLOAD_FOLDER) if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))]
    print(f"[presentation] starting with {len(files)} images")
    if not files:
        return {"ok": False, "error": "no images available"}
    start_presentation(files)
    return {"ok": True, "files": files}


@app.route("/manual_show/<image>", methods=["POST"])
def api_manual_show(image):
    manual_show(image)
    return {"ok": True, "image": image}

@app.route("/stop_presentation", methods=["POST"])
def api_stop_presentation():
    stop_presentation()
    return {"ok": True}

@app.route("/presentation/next", methods=["POST"])
def api_next_image():
    global current_index
    if not images_list:
        return {"ok": False, "error": "no images loaded"}, 400

    # move forward
    current_index = (current_index + 1) % len(images_list)
    image = images_list[current_index]

    manual_show(image)  # sends SHOW_IMAGE + pause slideshow
    return {"ok": True, "image": image}

@app.route("/presentation/prev", methods=["POST"])
def api_prev_image():
    global current_index
    if not images_list:
        return {"ok": False, "error": "no images loaded"}, 400

    # move backward
    current_index = (current_index - 1) % len(images_list)
    image = images_list[current_index]

    manual_show(image)  # sends SHOW_IMAGE + pause slideshow
    return {"ok": True, "image": image}
# enter calibration mode for client (client_id) and specific display (display_name)
@app.route("/calibrate/<client_id>/<display_name>", methods=["POST"])
def api_calibrate_display(client_id, display_name):
    sid = connected_clients.get(client_id)
    if not sid:
        return {"ok": False, "error": f"client {client_id} not connected"}, 400
    socketio.emit("CALIBRATE_DISPLAY", {"display_name": display_name}, room=sid)
    return {"ok": True, "client_id": client_id, "display_name": display_name}

#exit calibration mode for client (client_id) and specific display (display_name)
@app.route("/calibrate/<client_id>/<display_name>/exit", methods=["POST"])
def api_exit_calibrate_display(client_id, display_name):
    sid = connected_clients.get(client_id)
    if not sid:
        return {"ok": False, "error": f"client {client_id} not connected"}, 400
    socketio.emit("EXIT_CALIBRATE_DISPLAY", {"display_name": display_name}, room=sid)
    return {"ok": True, "client_id": client_id, "display_name": display_name}

# ---- SocketIO events for presentation sync ----

@socketio.on("PRESENTATION_READY")
def on_presentation_ready(data):
    cid = data.get("client_id")
    if data.get("ready"):
        clients_ready.add(cid)
        print(f"[server] {cid} is ready")

    if clients_ready == expected_clients:
        print("[server] All clients ready -> starting slideshow")
        slideshow_job = eventlet.spawn(slideshow_loop)
    return


@socketio.on("IMAGE_SHOWN")
def on_image_shown(data):
    cid = data.get("client_id")
    image = data.get("image")
    print(f"[server] Ack from {cid}: {image}")


# ---- Main ----
if __name__ == "__main__":
    SERVER_HOST = os.getenv("SERVER_HOST", "127.0.0.1")
    SERVER_PORT = int(os.getenv("SERVER_PORT", "5000"))
    print("Starting server on http://{}:{}".format(SERVER_HOST, SERVER_PORT))
    # socketio.run(app, host=SERVER_HOST, port=SERVER_PORT)
    eventlet.wsgi.server(eventlet.listen((SERVER_HOST, SERVER_PORT)), app)
    print("Server stopped")
