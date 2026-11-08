import sys
import cv2
import subprocess
import numpy as np
from insightface.app import FaceAnalysis
from db import (
    load_embeddings_matrix,
    save_person,
    add_fingerprint,
    find_person_by_name,
    clean_duplicates,
)


# --- Decision thresholds ---
AUTO_THRESHOLD = 0.30          # >=  => ask Y/n (Enter = Y), add fingerprint to this person
LOW_CONF_THRESHOLD = 0.10      # < AUTO_THRESHOLD but >= LOW => ask Y/n with warning (Enter = N)
UNKNOWN_THRESHOLD = 0.10       # <  => unknown #N, ask user for data
DUPLICATE_WARNING_THRESHOLD = 0.99  # >= => result 1.00, this is 100% the same person, do not add duplicate


def ask_confirmation(prompt: str, default_y: bool) -> bool:
    """
    Ask user Y/n.
    default_y=True  -> Enter means Y
    default_y=False -> Enter means N
    """
    suffix = "(Y/n)" if default_y else "(y/N)"
    ans = input(f"{prompt} {suffix}: ").strip().lower()
    if ans == "":
        return default_y
    return ans in ("y", "yes", "t", "tak")


def handle_manual_entry(index: int, embedding: np.ndarray, face_crop: np.ndarray):
    """
    Ask for name and surname for a given face index.
    Checks if such person already exists in the database:
    - if yes → adds a new fingerprint to the existing person
    - if no → creates a new person
    """
    user_input = input(f"data for #{index}: ").strip()
    if not user_input:
        print(f"   ⏭️  Skipped #{index}")
        return

    parts = user_input.split()
    name = parts[0] if len(parts) > 0 else ""
    surname = parts[1] if len(parts) > 1 else ""
    secondname = parts[2] if len(parts) > 2 else ""

    if not name and not surname:
        print(f"   ⏭️  Skipped #{index} (no name and surname)")
        return

    existing_id = find_person_by_name(name, surname)
    if existing_id:
        new_idx = add_fingerprint(existing_id, embedding, face_crop,
                                  check_duplicates=True)
        if new_idx is not None:
            print(f"   ✅ Added new fingerprint to existing person {name} {surname} (index {new_idx})")
        else:
            print(f"   ⛔ Blocked as duplicate (already exists in database)")
    else:
        person_id = save_person(
            name=name,
            surname=surname,
            secondname=secondname,
            embedding=embedding,
            face_image_bgr=face_crop,
        )
        print(f"   ✅ Created new person {name} {surname} {secondname}: {person_id}")


def recognize_from_file(image_path: str, tolerance: float = 0.4):
    """
    Recognizes faces in an image and allows interactively adding new fingerprints
    to existing persons or creating new persons from unknown faces.

    `tolerance` is kept for CLI compatibility,
    but internally the thresholds above are used.
    """
    # Automatic duplicate cleanup on start
    removed = clean_duplicates()
    if removed:
        print(f"🧹 Removed {removed} duplicate fingerprint(s) from database.\n")

    app = FaceAnalysis(name="buffalo_l", root=".", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Cannot load image: {image_path}")
        return

    faces = app.get(img)
    if not faces:
        print("❌ No faces detected in the image.")
        return

    print(f"Detected {len(faces)} face(s).\n")

    # All fingerprints from the database in one matrix - fast batch cosine similarity
    matrix, meta = load_embeddings_matrix()
    has_db = len(matrix) > 0

    out = img.copy()
    results = []  # results for each entry (face)

    for i, face in enumerate(faces):
        x1, y1, x2, y2 = face.bbox[:4].astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(out.shape[1], x2), min(out.shape[0], y2)

        face_crop = img[y1:y2, x1:x2]
        emb = face.normed_embedding.astype(np.float32)

        best_sim = -1.0
        best_meta = None

        if has_db:
            # cosine similarity: all vectors are already normalized by insightface
            sims = matrix @ emb  # shape (N,)
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            best_meta = meta[best_idx]

        entry = {
            "index": i,
            "bbox": (x1, y1, x2, y2),
            "face_crop": face_crop,
            "embedding": emb,
        }

        if best_sim >= DUPLICATE_WARNING_THRESHOLD and best_meta is not None:
            info = best_meta["info"]
            label = f"#{i}: {info['name']} {info['surname']} ({best_sim:.2f}) [duplicate]"
            color = (0, 255, 0)  # green - correct result, this is 100% the same person
            entry["status"] = "duplicate_warning"
            entry["person_id"] = best_meta["person_id"]
            entry["sim"] = best_sim
            entry["candidate_name"] = f"{info['name']} {info['surname']}"
            print(f"✅ #{i}: {info['name']} {info['surname']} ({best_sim:.2f}) – duplicate, this is 100% the same person")

        elif best_sim >= AUTO_THRESHOLD and best_meta is not None:
            info = best_meta["info"]
            label = f"#{i}: {info['name']} {info['surname']} ({best_sim:.2f})"
            color = (0, 255, 0)  # green
            entry["status"] = "confirm_high"
            entry["person_id"] = best_meta["person_id"]
            entry["sim"] = best_sim
            entry["candidate_name"] = f"{info['name']} {info['surname']}"
            print(f"✅ {label}")

        elif best_sim >= LOW_CONF_THRESHOLD and best_meta is not None:
            info = best_meta["info"]
            label = f"#{i}: ? {info['name']} {info['surname']} ({best_sim:.2f})"
            color = (0, 255, 255)  # yellow
            entry["status"] = "confirm_low"
            entry["person_id"] = best_meta["person_id"]
            entry["sim"] = best_sim
            entry["candidate_name"] = f"{info['name']} {info['surname']}"
            print(f"⚠️  {label}")

        else:
            label = f"#{i}: unknown #{i}"
            color = (0, 0, 255)  # red
            entry["status"] = "unknown"
            entry["unknown_index"] = i
            print(f"❌ {label}")

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, label, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        results.append(entry)

    cv2.imwrite("result.jpg", out)
    print("\nResult with boxes saved to result.jpg")

    # Open result with eog (Eye of GNOME) if available
    try:
        subprocess.Popen(["eog", "result.jpg"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("eog not found – open result.jpg with your own image viewer.")

    # --- Interactive confirmation / adding new people ---
    for entry in results:
        if entry["status"] == "duplicate_warning":
            # Does not ask, does not add fingerprint - only informs
            continue

        elif entry["status"] == "confirm_high":
            pid = entry["person_id"]
            cand = entry["candidate_name"]
            sim = entry["sim"]
            if ask_confirmation(f"Is #{entry['index']} {cand} ({sim:.2f})?", default_y=True):
                new_idx = add_fingerprint(pid, entry["embedding"], entry["face_crop"],
                                          check_duplicates=True)
                if new_idx is not None:
                    print(f"   ✅ Added new fingerprint to {cand} (index {new_idx})")
                else:
                    print(f"   ⛔ Blocked as duplicate (already exists in database)")
            else:
                handle_manual_entry(entry["index"], entry["embedding"], entry["face_crop"])

        elif entry["status"] == "confirm_low":
            pid = entry["person_id"]
            cand = entry["candidate_name"]
            sim = entry["sim"]
            print(f"⚠️  Confidence was not high ({sim:.2f}).")
            if ask_confirmation(f"Is #{entry['index']} {cand}?", default_y=False):
                new_idx = add_fingerprint(pid, entry["embedding"], entry["face_crop"],
                                          check_duplicates=True)
                if new_idx is not None:
                    print(f"   ✅ Added new fingerprint to {cand} (index {new_idx})")
                else:
                    print(f"   ⛔ Blocked as duplicate (already exists in database)")
            else:
                handle_manual_entry(entry["index"], entry["embedding"], entry["face_crop"])

        elif entry["status"] == "unknown":
            handle_manual_entry(entry["unknown_index"], entry["embedding"], entry["face_crop"])


if __name__ == "__main__":
    tol = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
    recognize_from_file(sys.argv[1], tol)
