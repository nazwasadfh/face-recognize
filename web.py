#!/usr/bin/env python3
"""
face-web: Flask web panel for FaceSearch database.
A hacker-style police database interface for browsing persons,
searching records, and running face recognition on uploaded images.
"""

import os
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory, send_file

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from db import load_embeddings_matrix, PERSONS_DIR

app = Flask(__name__, static_folder="static", static_url_path="/static")

_face_app = None


def get_face_app():
    """Lazy-initialize FaceAnalysis (loads models once)."""
    global _face_app
    if _face_app is None:
        _face_app = FaceAnalysis(
            name="buffalo_l",
            root=".",
            providers=["CPUExecutionProvider"],
        )
        _face_app.prepare(ctx_id=0, det_size=(640, 640))
    return _face_app


def list_persons():
    """Return list of person dicts with basic metadata."""
    persons = []
    for person_dir in PERSONS_DIR.iterdir():
        if not person_dir.is_dir():
            continue
        info_path = person_dir / "info.json"
        if not info_path.exists():
            continue
        try:
            with open(info_path, encoding="utf-8") as f:
                info = json.load(f)
        except (OSError, ValueError):
            continue

        fp_dir = person_dir / "fingerprints"
        fp_count = 0
        if fp_dir.exists():
            fp_count = len([p for p in fp_dir.glob("*.npy") if p.stem.isdigit()])

        persons.append({
            "person_id": person_dir.name,
            "name": info.get("name", ""),
            "surname": info.get("surname", ""),
            "secondname": info.get("secondname", ""),
            "fingerprint_count": fp_count,
            "imported_count": len(info.get("imported_from_input", [])),
            "photo_url": f"/photo/{person_dir.name}",
        })
    return persons


import json  # noqa: E402


@app.route("/")
def index():
    return send_file("static/index.html")


@app.route("/api/persons")
def api_persons():
    """List all persons in the database."""
    return jsonify(list_persons())


@app.route("/api/persons/<person_id>")
def api_person_detail(person_id):
    """Return full info.json + metadata for a single person."""
    person_dir = PERSONS_DIR / person_id
    info_path = person_dir / "info.json"
    if not info_path.exists():
        return jsonify({"error": "not found"}), 404

    with open(info_path, encoding="utf-8") as f:
        info = json.load(f)

    fp_dir = person_dir / "fingerprints"
    fingerprints = []
    if fp_dir.exists():
        for p in sorted(fp_dir.glob("*.npy"), key=lambda x: int(x.stem) if x.stem.isdigit() else 0):
            fingerprints.append({"index": int(p.stem), "url": f"/fingerprint/{person_id}/{p.stem}"})

    return jsonify({
        "person_id": person_id,
        "info": info,
        "fingerprints": fingerprints,
        "photo_url": f"/photo/{person_id}",
    })


@app.route("/api/search")
def api_search():
    """Partial case-insensitive search by name/surname/secondname."""
    q = request.args.get("q", "").strip().lower()
    if not q:
        return jsonify([])

    results = []
    for p in list_persons():
        haystack = f"{p['name']} {p['surname']} {p['secondname']}".lower()
        if q in haystack:
            results.append(p)
    return jsonify(results)


@app.route("/photo/<person_id>")
def photo(person_id):
    """Serve the cropped face photo of a person."""
    person_dir = PERSONS_DIR / person_id
    photo_path = person_dir / "photo.png"
    if not photo_path.exists():
        return jsonify({"error": "no photo"}), 404
    return send_from_directory(person_dir, "photo.png")


@app.route("/fingerprint/<person_id>/<fp_name>")
def fingerprint(person_id, fp_name):
    """Serve a fingerprint embedding as raw bytes (mostly for debugging)."""
    person_dir = PERSONS_DIR / person_id
    fp_path = person_dir / "fingerprints" / f"{fp_name}.npy"
    if not fp_path.exists():
        return jsonify({"error": "not found"}), 404
    return send_from_directory(person_dir / "fingerprints", f"{fp_name}.npy")


@app.route("/api/recognize", methods=["POST"])
def api_recognize():
    """
    Accept an image file, detect faces, compare against the database,
    and return matches with similarity scores.
    """
    if "image" not in request.files:
        return jsonify({"error": "no image provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "empty filename"}), 400

    tmp_name = f"webscan_{os.getpid()}_{uuid.uuid4().hex[:8]}.jpg"
    tmp_path = Path("/tmp") / tmp_name
    file.save(str(tmp_path))

    try:
        face_app = get_face_app()
        img = cv2.imread(str(tmp_path))
        if img is None:
            return jsonify({"error": "cannot read image"}), 400

        faces = face_app.get(img)
        matrix, meta = load_embeddings_matrix()
        has_db = len(matrix) > 0

        results = []
        for face in faces:
            emb = face.normed_embedding.astype(np.float32)

            best_sim = -1.0
            best_meta = None
            if has_db:
                sims = matrix @ emb
                best_idx = int(np.argmax(sims))
                best_sim = float(sims[best_idx])
                best_meta = meta[best_idx]

            person = None
            if best_meta is not None and best_sim >= 0.30:
                info = best_meta["info"]
                person = {
                    "person_id": best_meta["person_id"],
                    "name": info.get("name", ""),
                    "surname": info.get("surname", ""),
                    "secondname": info.get("secondname", ""),
                    "similarity": round(best_sim, 3),
                }

            results.append({
                "bbox": [int(v) for v in face.bbox[:4]],
                "person": person,
                "similarity": round(best_sim, 3) if best_meta else None,
            })

        return jsonify({"results": results, "image_url": f"/tmp-scan/{tmp_name}"})
    finally:
        tmp_path.unlink(missing_ok=True)


@app.route("/tmp-scan/<filename>")
def tmp_scan(filename):
    """Serve a temporarily uploaded scan image."""
    return send_from_directory("/tmp", filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
