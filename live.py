import sys
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from db import load_all_persons


def main():
    tolerance = float(sys.argv[1]) if len(sys.argv) > 1 else 0.4
    camera_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    persons = load_all_persons()
    print(f"Wczytano {len(persons)} osób z bazy.")
    if not persons:
        print("⚠️  Baza jest pusta – wszystkie twarze będą oznaczane jako nieznane.")

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"❌ Nie można otworzyć kamery {camera_id}")
        return

    print("Naciśnij 'q' lub ESC, aby wyjść z podglądu na żywo.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Brak klatki z kamery.")
            break

        faces = app.get(frame)

        out = frame.copy()
        for face in faces:
            x1, y1, x2, y2 = face.bbox[:4].astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(out.shape[1], x2), min(out.shape[0], y2)

            # Znajdź najbliższą osobę w bazie (cosine similarity)
            best_sim = -1.0
            best_person = None
            for p in persons:
                sim = float(np.dot(p["embedding"], face.normed_embedding))
                if sim > best_sim:
                    best_sim = sim
                    best_person = p

            if best_person and best_sim >= tolerance:
                info = best_person["info"]
                label = f"{info['name']} {info['surname']} ({best_sim:.2f})"
                color = (0, 255, 0)  # zielony – rozpoznano
            else:
                label = f"Nieznana osoba ({best_sim:.2f})"
                color = (0, 0, 255)  # czerwony – nieznana

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, label, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("FaceSearch - Live Preview", out)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q lub ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
