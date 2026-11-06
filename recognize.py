import sys
import cv2
import subprocess
import numpy as np
from insightface.app import FaceAnalysis
from db import (
    load_embeddings_matrix,
    save_person,
    add_fingerprint,
    clean_duplicates,
)


# --- Progi decyzyjne ---
AUTO_THRESHOLD = 0.30          # >=  => pyta Y/n (enter = Y), dodaje fingerprint do tej osoby
LOW_CONF_THRESHOLD = 0.10      # < AUTO_THRESHOLD ale >= LOW => pyta Y/n z ostrzeżeniem (enter = N)
UNKNOWN_THRESHOLD = 0.10       # <  => unknown #N, prosi o dane od użytkownika
DUPLICATE_WARNING_THRESHOLD = 0.99  # >= => wynik 1.00, to na 100% ta sama osoba, nie dodaje duplikatu


def ask_confirmation(prompt: str, default_y: bool) -> bool:
    """
    Pyta użytkownika Y/n.
    default_y=True  -> enter oznacza Y
    default_y=False -> enter oznacza N
    """
    suffix = "(Y/n)" if default_y else "(y/N)"
    ans = input(f"{prompt} {suffix}: ").strip().lower()
    if ans == "":
        return default_y
    return ans in ("y", "yes", "t", "tak")


def recognize_from_file(image_path: str, tolerance: float = 0.4):
    """
    Rozpoznaje twarze na obrazie i pozwala interaktywnie dodać nowe fingerprinti
    do istniejących osób lub utworzyć nowe osoby z nieznanych twarzy.

    `tolerance` jest zachowane dla kompatybilności wywołania CLI,
    ale wewnątrz używane są powyższe progi.
    """
    # Automatyczne czyszczenie duplikatów przy starcie
    removed = clean_duplicates()
    if removed:
        print(f"🧹 Usunięto {removed} duplikat(ów) fingerprintów z bazy.\n")

    app = FaceAnalysis(name="buffalo_l", root=".", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Nie można wczytać obrazu: {image_path}")
        return

    faces = app.get(img)
    if not faces:
        print("❌ Nie wykryto twarzy na zdjęciu.")
        return

    print(f"Wykryto {len(faces)} twarz/twarze.\n")

    # Wszystkie fingerprints z bazy w jednej macierzy – szybkie batch cosine similarity
    matrix, meta = load_embeddings_matrix()
    has_db = len(matrix) > 0

    out = img.copy()
    results = []  # wyniki dla każdego entry (twarzy)

    for i, face in enumerate(faces):
        x1, y1, x2, y2 = face.bbox[:4].astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(out.shape[1], x2), min(out.shape[0], y2)

        face_crop = img[y1:y2, x1:x2]
        emb = face.normed_embedding.astype(np.float32)

        best_sim = -1.0
        best_meta = None

        if has_db:
            # cosine similarity: wszystkie wektory są już znormalizowane przez insightface
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
            label = f"#{i}: {info['name']} {info['surname']} ({best_sim:.2f}) [duplikat]"
            color = (0, 255, 0)  # zielony – wynik poprawny, to na 100% ta sama osoba
            entry["status"] = "duplicate_warning"
            entry["person_id"] = best_meta["person_id"]
            entry["sim"] = best_sim
            entry["candidate_name"] = f"{info['name']} {info['surname']}"
            print(f"✅ #{i}: {info['name']} {info['surname']} ({best_sim:.2f}) – duplikat, to na 100% ta sama osoba")

        elif best_sim >= AUTO_THRESHOLD and best_meta is not None:
            info = best_meta["info"]
            label = f"#{i}: {info['name']} {info['surname']} ({best_sim:.2f})"
            color = (0, 255, 0)  # zielony
            entry["status"] = "confirm_high"
            entry["person_id"] = best_meta["person_id"]
            entry["sim"] = best_sim
            entry["candidate_name"] = f"{info['name']} {info['surname']}"
            print(f"✅ {label}")

        elif best_sim >= LOW_CONF_THRESHOLD and best_meta is not None:
            info = best_meta["info"]
            label = f"#{i}: ? {info['name']} {info['surname']} ({best_sim:.2f})"
            color = (0, 255, 255)  # żółty
            entry["status"] = "confirm_low"
            entry["person_id"] = best_meta["person_id"]
            entry["sim"] = best_sim
            entry["candidate_name"] = f"{info['name']} {info['surname']}"
            print(f"⚠️  {label}")

        else:
            label = f"#{i}: unknown #{i}"
            color = (0, 0, 255)  # czerwony
            entry["status"] = "unknown"
            entry["unknown_index"] = i
            print(f"❌ {label}")

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, label, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        results.append(entry)

    cv2.imwrite("result.jpg", out)
    print("\nWynik z ramkami zapisano do result.jpg")

    # Otwórz wynik w eog (Eye of GNOME), jeśli dostępne
    try:
        subprocess.Popen(["eog", "result.jpg"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("Nie znaleziono eog – otwórz result.jpg własnym programem do obrazów.")

    # --- Interaktywne potwierdzenie / dodawanie nowych osób ---
    for entry in results:
        if entry["status"] == "duplicate_warning":
            # Nie pyta, nie dodaje fingerprinta – tylko informuje
            continue

        elif entry["status"] == "confirm_high":
            pid = entry["person_id"]
            cand = entry["candidate_name"]
            sim = entry["sim"]
            if ask_confirmation(f"Czy #{entry['index']} to {cand} ({sim:.2f})?", default_y=True):
                new_idx = add_fingerprint(pid, entry["embedding"], entry["face_crop"],
                                          check_duplicates=True)
                if new_idx is not None:
                    print(f"   ✅ Dodano nowy fingerprint do {cand} (indeks {new_idx})")
                else:
                    print(f"   ⛔ Zablokowano jako duplikat (już istnieje w bazie)")
            else:
                print(f"   ⏭️  Pominięto #{entry['index']}")

        elif entry["status"] == "confirm_low":
            pid = entry["person_id"]
            cand = entry["candidate_name"]
            sim = entry["sim"]
            print(f"⚠️  Wynik nie był wysoki ({sim:.2f}).")
            if ask_confirmation(f"Czy #{entry['index']} to {cand}?", default_y=False):
                new_idx = add_fingerprint(pid, entry["embedding"], entry["face_crop"],
                                          check_duplicates=True)
                if new_idx is not None:
                    print(f"   ✅ Dodano nowy fingerprint do {cand} (indeks {new_idx})")
                else:
                    print(f"   ⛔ Zablokowano jako duplikat (już istnieje w bazie)")
            else:
                print(f"   ⏭️  Pominięto #{entry['index']}")

        elif entry["status"] == "unknown":
            uid = entry["unknown_index"]
            user_input = input(f"data for #{uid}: ").strip()
            if not user_input:
                print(f"   ⏭️  Pominięto unknown #{uid}")
                continue

            parts = user_input.split()
            name = parts[0] if len(parts) > 0 else ""
            surname = parts[1] if len(parts) > 1 else ""
            secondname = parts[2] if len(parts) > 2 else ""

            if not name and not surname:
                print(f"   ⏭️  Pominięto unknown #{uid} (brak imienia i nazwiska)")
                continue

            person_id = save_person(
                name=name,
                surname=surname,
                secondname=secondname,
                embedding=entry["embedding"],
                face_image_bgr=entry["face_crop"],
            )
            print(f"   ✅ Zapisano unknown #{uid} jako nową osobę: {person_id} ({name} {surname} {secondname})")


if __name__ == "__main__":
    tol = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
    recognize_from_file(sys.argv[1], tol)
