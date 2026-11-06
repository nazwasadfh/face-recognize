import sys
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from db import load_embeddings_matrix


# --- Progi decyzyjne (takie same jak w recognize.py) ---
DUPLICATE_WARNING_THRESHOLD = 0.99  # >= => na 100% ta sama osoba, zielony
AUTO_THRESHOLD = 0.30               # >=  => rozpoznane, zielony
LOW_CONF_THRESHOLD = 0.10           # >= i < AUTO => możliwe, żółty
UNKNOWN_THRESHOLD = 0.10            # < => nieznana, czerwony


def main():
    tolerance = float(sys.argv[1]) if len(sys.argv) > 1 else 0.4
    camera_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    process_every = int(sys.argv[3]) if len(sys.argv) > 3 else 2  # co ile klatek przetwarzać

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(320, 320))  # mniejsze = szybciej

    # Wczytanie wszystkich fingerprintów do jednej macierzy – bez pętli po osobach
    matrix, meta = load_embeddings_matrix()
    has_db = len(matrix) > 0
    print(f"Wczytano {len(meta)} fingerprintów z bazy.")
    print(f"Przetwarzanie co {process_every} klatki, det_size=320x320")

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"❌ Nie można otworzyć kamery {camera_id}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Naciśnij 'q' lub ESC, aby wyjść z podglądu na żywo.")

    frame_count = 0
    cached_results = []  # (x1, y1, x2, y2, label, color)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Brak klatki z kamery.")
            break

        # Co N klatek wykonaj ciężką detekcję + rozpoznawanie batch
        if frame_count % process_every == 0:
            faces = app.get(frame)
            cached_results = []
            for face in faces:
                x1, y1, x2, y2 = face.bbox[:4].astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                emb = face.normed_embedding.astype(np.float32)

                best_sim = -1.0
                best_meta = None

                if has_db:
                    # cosine similarity dla wszystkich fingerprintów naraz (macierzowo)
                    sims = matrix @ emb  # shape (N,)
                    best_idx = int(np.argmax(sims))
                    best_sim = float(sims[best_idx])
                    best_meta = meta[best_idx]

                if best_sim >= DUPLICATE_WARNING_THRESHOLD and best_meta is not None:
                    info = best_meta["info"]
                    label = f"#{frame_count % 1000}: {info['name']} {info['surname']} ({best_sim:.2f}) [duplikat]"
                    color = (0, 255, 0)  # zielony
                elif best_sim >= AUTO_THRESHOLD and best_meta is not None:
                    info = best_meta["info"]
                    label = f"#{frame_count % 1000}: {info['name']} {info['surname']} ({best_sim:.2f})"
                    color = (0, 255, 0)  # zielony
                elif best_sim >= LOW_CONF_THRESHOLD and best_meta is not None:
                    info = best_meta["info"]
                    label = f"#{frame_count % 1000}: ? {info['name']} {info['surname']} ({best_sim:.2f})"
                    color = (0, 255, 255)  # żółty
                else:
                    label = f"#{frame_count % 1000}: unknown"
                    color = (0, 0, 255)  # czerwony

                cached_results.append((x1, y1, x2, y2, label, color))

        # Rysuj buforowane wyniki na bieżącej klatce (bardzo szybko)
        out = frame.copy()
        for (x1, y1, x2, y2, label, color) in cached_results:
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, label, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(out, f"co {process_every} klatki | q/ESC = wyjscie", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("FaceSearch - Live Preview", out)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q lub ESC
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
