#!/usr/bin/env python3
"""Talking Faces studio — local web app.

Upload phoneme sprite sheets (stored in a library), then render talking-face
videos either from typed text (TTS) or from an audio file + timed lyrics.

Run:  .venv/bin/python webapp/app.py   (serves http://localhost:5173)
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from pumpkin import FPS, STYLES, envelope, load_audio, tts
from sprite import lyric_tokens, render_video, slice_sheet, slice_sheet_color, sprite_track

SHEETS = os.path.join(HERE, "sheets")
RENDERS = os.path.join(HERE, "renders")
UPLOADS = os.path.join(HERE, "uploads")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

JOBS = {}  # id -> {status, progress, error, url, label}


# ---------- sheets library ----------

@app.get("/api/sheets")
def list_sheets():
    out = []
    for f in sorted(os.listdir(SHEETS)):
        if f.endswith(".json"):
            with open(os.path.join(SHEETS, f)) as fh:
                out.append(json.load(fh))
    return jsonify(out)


@app.post("/api/sheets")
def upload_sheet():
    file = request.files.get("file")
    name = request.form.get("name") or "Untitled"
    if not file:
        return jsonify({"error": "no file"}), 400
    sid = uuid.uuid4().hex[:10]
    ext = os.path.splitext(file.filename or "x.png")[1].lower() or ".png"
    path = os.path.join(SHEETS, sid + ext)
    file.save(path)
    try:  # validate: must slice as a color sheet (connected art) or ink grid
        slice_sheet_color(path)
    except Exception:
        try:
            slice_sheet(path)
        except Exception as e:
            os.unlink(path)
            return jsonify({"error": f"couldn't read a 3x3 sprite grid: {e}"}), 400
    meta = {"id": sid, "name": name, "file": os.path.basename(path)}
    with open(os.path.join(SHEETS, sid + ".json"), "w") as fh:
        json.dump(meta, fh)
    return jsonify(meta)


@app.patch("/api/sheets/<sid>")
def rename_sheet(sid):
    meta_path = os.path.join(SHEETS, sid + ".json")
    if not os.path.exists(meta_path):
        return jsonify({"error": "not found"}), 404
    with open(meta_path) as fh:
        meta = json.load(fh)
    meta["name"] = (request.get_json() or {}).get("name", meta["name"]).strip() or meta["name"]
    with open(meta_path, "w") as fh:
        json.dump(meta, fh)
    return jsonify(meta)


@app.delete("/api/sheets/<sid>")
def delete_sheet(sid):
    for f in os.listdir(SHEETS):
        if f.startswith(sid + "."):
            os.unlink(os.path.join(SHEETS, f))
    return jsonify({"ok": True})


@app.get("/sheets/<path:fn>")
def sheet_file(fn):
    return send_from_directory(SHEETS, fn)


# ---------- rendering ----------

def sheet_path(sid):
    with open(os.path.join(SHEETS, sid + ".json")) as fh:
        return os.path.join(SHEETS, json.load(fh)["file"])


def run_job(jid, fn):
    def wrap():
        try:
            fn()
            JOBS[jid]["status"] = "done"
        except Exception as e:
            JOBS[jid]["status"] = "error"
            JOBS[jid]["error"] = str(e)
    threading.Thread(target=wrap, daemon=True).start()


def make_progress(jid):
    def cb(done, total):
        JOBS[jid]["progress"] = round(done / total * 100)
    return cb


@app.post("/api/render/tts")
def render_tts():
    d = request.get_json()
    spath = sheet_path(d["sheet"])
    style = d.get("style", "pumpkin")
    W, H = (int(v) for v in d.get("size", "1280x720").split("x"))
    text, voice = d["text"], d.get("voice", "am_michael")
    jid = uuid.uuid4().hex[:10]
    out = os.path.join(RENDERS, f"{jid}.mp4")
    JOBS[jid] = {"status": "running", "progress": 0, "label": text[:60],
                 "url": f"/renders/{jid}.mp4", "created": time.time()}

    def work():
        wav = tempfile.mktemp(suffix=".wav")
        _, tokens = tts(text, voice, wav)
        audio, sr = load_audio(wav)
        n = max(1, int(len(audio) / sr * FPS))
        env = envelope(audio, sr, n)
        track = sprite_track(tokens, env, n)
        render_video(spath, track, env, wav, out, style, W, H, make_progress(jid))
        os.unlink(wav)

    run_job(jid, work)
    return jsonify({"job": jid})


@app.post("/api/render/song")
def render_song():
    spath = sheet_path(request.form["sheet"])
    style = request.form.get("style", "pumpkin")
    W, H = (int(v) for v in request.form.get("size", "1280x720").split("x"))
    lrc = request.form["lyrics"]
    audio_file = request.files.get("audio")
    if not audio_file:
        return jsonify({"error": "no audio file"}), 400
    raw = os.path.join(UPLOADS, uuid.uuid4().hex[:10] + os.path.splitext(audio_file.filename or "a")[1])
    audio_file.save(raw)
    wav = raw + ".wav"  # normalize anything (mp3/m4a/...) to wav
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", raw, "-ac", "1", wav],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return jsonify({"error": "couldn't decode audio: " + r.stderr[:200]}), 400

    jid = uuid.uuid4().hex[:10]
    out = os.path.join(RENDERS, f"{jid}.mp4")
    label = (lrc.strip().splitlines() or ["song"])[0][:60]
    JOBS[jid] = {"status": "running", "progress": 0, "label": label,
                 "url": f"/renders/{jid}.mp4", "created": time.time()}

    def work():
        audio, sr = load_audio(wav)
        dur = len(audio) / sr
        n = max(1, int(dur * FPS))
        env = envelope(audio, sr, n)
        tokens = lyric_tokens(lrc, dur)
        if not tokens:
            raise RuntimeError("no timed lyric lines found — use [mm:ss.xx] Lyric text")
        track = sprite_track(tokens, env, n)
        render_video(spath, track, env, wav, out, style, W, H, make_progress(jid))

    run_job(jid, work)
    return jsonify({"job": jid})


@app.get("/api/jobs/<jid>")
def job_status(jid):
    if jid in JOBS:
        return jsonify(JOBS[jid])
    # server restarted mid-poll: report done if the file made it to disk
    if os.path.exists(os.path.join(RENDERS, f"{jid}.mp4")):
        return jsonify({"status": "done", "progress": 100, "url": f"/renders/{jid}.mp4"})
    return jsonify({"status": "error", "error": "job lost (server restarted) — render again"})


@app.get("/api/renders")
def list_renders():
    out = []
    for f in sorted(os.listdir(RENDERS), key=lambda f: -os.path.getmtime(os.path.join(RENDERS, f))):
        if f.endswith(".mp4"):
            jid = f[:-4]
            meta = JOBS.get(jid, {})
            out.append({"url": f"/renders/{f}", "label": meta.get("label", f),
                        "mtime": os.path.getmtime(os.path.join(RENDERS, f))})
    return jsonify(out)


@app.get("/renders/<path:fn>")
def render_file(fn):
    return send_from_directory(RENDERS, fn)


@app.get("/")
def index():
    return send_from_directory(HERE, "index.html")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5173, debug=False)
