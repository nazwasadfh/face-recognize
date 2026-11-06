import sys
import cv2
import numpy as np
import threading
import time
from insightface.app import FaceAnalysis
from db import load_embeddings_matrix


# --- Progi decyzyjne (takie same jak w recognize.py) ---
DUPLICATE_WARNING_THRESHOLD = 0.99  # >= => na 100% ta sama osoba, zielony
AUTO_THRESHOLD = 0.30               # >=  => rozpoznane, zielony
LOW_CONF_THRESHOLD = 0.10           # >= i < AUTO => możliwe, żółty
UNKNOWN_THRESHOLD = 0.10            # < => nieznana, czerwony


class SharedState:
    """Dzielona pamięć między wątkiem kamery, przetwarzania i wyświetlania."""
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None       # najświeższa klatka z kamery
        self.results = []       # ostatnie wyniki rozpoznawania: (x1,y1,x2,y2,label,color)
        self.running = True


def camera_loop(cap: cv2.VideoCapture, state: SharedState):
    """Ciągle pobiera najnowszą klatkę z kamery – nigdy nie czeka na przetwarzanie."""
    while state.running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.005)
            continue
        with state.lock:
            state.frame = frame
        time.sleep(0.001)


def process_loop(state: SharedState, app: FaceAnalysis,
                 matrix: np.ndarray, meta: list, process_every: int):
    """W tle wykonuje detekcję i rozpoznawanie na najświeższej klatce."""
    has_db = len(matrix) > 0
    counter = 0

    while state.running:
        with state.lock:
            frame = None if state.frame is None else state.frame.copy()
        if frame is None:
            time.sleep(0.005)
            continue

        counter += 1
        if counter % process_every != 0:
            time.sleep(0.001)
            continue

        faces = app.get(frame)
        results = []

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
                label = f"{info['name']} {info['surname']} ({best_sim:.2f}) [duplikat]"
                color = (0, 255, 0)  # zielony
            elif best_sim >= AUTO_THRESHOLD and best_meta is not None:
                info = best_meta["info"]
                label = f"{info['name']} {info['surname']} ({best_sim:.2f})"
                color = (0, 255, 0)  # zielony
            elif best_sim >= LOW_CONF_THRESHOLD and best_meta is not None:
                info = best_meta["info"]
                label = f"? {info['name']} {info['surname']} ({best_sim:.2f})"
                color = (0, 255, 255)  # żółty
            else:
                label = "unknown"
                color = (0, 0, 255)  # czerwony

            results.append((x1, y1, x2, y2, label, color))

        with state.lock:
            state.results = results
        time.sleep(0.001)


def main():
    tolerance = float(sys.argv[1]) if len(sys.argv) > 1 else 0.4
    camera_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    process_every = int(sys.argv[3]) if len(sys.argv) > 3 else 2

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(320, 320))

    matrix, meta = load_embeddings_matrix()
    print(f"Wczytano {len(meta)} fingerprintów z bazy.")

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"❌ Nie można otworzyć kamery {camera_id}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    state = SharedState()

    t_cam = threading.Thread(target=camera_loop, args=(cap, state), daemon=True)
    t_proc = threading.Thread(target=process_loop, args=(state, app, matrix, meta, process_every), daemon=True)
    t_cam.start()
    t_proc.start()

    print("Naciśnij 'q' lub ESC, aby wyjść.")

    try:
        while state.running:
            with state.lock:
                frame = None if state.frame is None else state.frame.copy()
                results = list(state.results)

            if frame is None:
                time.sleep(0.005)
                continue

            out = frame.copy()
            for (x1, y1, x2, y2, label, color) in results:
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                cv2.putText(out, label, (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.putText(out, f"co {process_every} klatki | q/ESC = wyjscie", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("FaceSearch - Live Preview", out)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        state.running = False
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
