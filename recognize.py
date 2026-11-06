import sys
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from db import load_embeddings_matrix, save_person, add_fingerprint


# --- Progi decyzyjne (dostosuj do swoich potrzeb) ---
AUTO_CONFIRM = 0.75   # >=  => automatycznie rozpoznane (zielona ramka)
ASK_THRESHOLD = 0.50  # >= i < AUTO_CONFIRM => pytaj o potwierdzenie (żółta ramka)
# poniżej ASK_THRESHOLD => unknown (czerwona ramka)


def recognize_from_file(image_path: str, tolerance: float = 0.4):
    """
    Rozpoznaje twarze na obrazie.

    - tolerance jest zachowane dla kompatybilności wywołania CLI,
      ale wewnątrz używane są powyższe progi AUTO_CONFIRM / ASK_THRESHOLD.
    """
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
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
    unknown_counter = 0

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

        if best_sim >= AUTO_CONFIRM and best_meta is not None:
            info = best_meta["info"]
            label = f"#{i}: {info['name']} {info['surname']} ({best_sim:.2f})"
            color = (0, 255, 0)  # zielony
            entry["status"] = "confirmed"
            entry["person_id"] = best_meta["person_id"]
            entry["sim"] = best_sim
            print(f"✅ {label}")

        elif best_sim >= ASK_THRESHOLD and best_meta is not None:
            info = best_meta["info"]
            label = f"#{i}: ? {info['name']} {info['surname']} ({best_sim:.2f})"
            color = (0, 255, 255)  # żółty
            entry["status"] = "ask"
            entry["person_id"] = best_meta["person_id"]
            entry["sim"] = best_sim
            entry["candidate_name"] = f"{info['name']} {info['surname']}"
            print(f"❓ {label}")

        else:
            label = f"#{i}: unknown #{unknown_counter}"
            color = (0, 0, 255)  # czerwony
            entry["status"] = "unknown"
            entry["unknown_index"] = unknown_counter
            unknown_counter += 1
            print(f"❌ {label}")

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, label, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        results.append(entry)

    cv2.imwrite("result.jpg", out)
    print("\nWynik z ramkami zapisano do result.jpg")

    # --- Interaktywne potwierdzenie / dodawanie nowych osób ---
    for entry in results:
        if entry["status"] == "ask":
            pid = entry["person_id"]
            cand = entry["candidate_name"]
            ans = input(f"Czy #{entry['index']} to {cand}? (t/n): ").strip().lower()
            if ans in ("t", "tak", "y", "yes"):
                add_fingerprint(pid, entry["embedding"], entry["face_crop"])
                print(f"   ✅ Dodano nowy fingerprint do {cand}")
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

    cv2.imshow("Rozpoznawanie twarzy", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    tol = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
    recognize_from_file(sys.argv[1], tol)
