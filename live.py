import sys
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from db import load_all_persons


def main():
    tolerance = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
    camera_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    process_every = (
        int(sys.argv[3]) if len(sys.argv) > 3 else 20
    )  # co ile klatek przetwarzać

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(320, 320))  # mniejsze = szybciej

    persons = load_all_persons()
    print(f"Wczytano {len(persons)} osób z bazy.")
    print(f"Przetwarzanie co {process_every} klatki, det_size=320x320")

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"❌ Nie można otworzyć kamery {camera_id}")
        return

    # Zmniejsz rozdzielczość kamery dla szybszego działania
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Naciśnij 'q' lub ESC, aby wyjść.")

    frame_count = 0
    cached_results = []  # (x1, y1, x2, y2, label, color)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Brak klatki z kamery.")
            break

        # Co N klatek wykonaj ciężką detekcję + rozpoznawanie
        if frame_count % process_every == 0:
            faces = app.get(frame)
            cached_results = []
            for face in faces:
                x1, y1, x2, y2 = face.bbox[:4].astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

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
                    color = (0, 255, 0)  # zielony
                else:
                    label = f"Nieznana osoba ({best_sim:.2f})"
                    color = (0, 0, 255)  # czerwony

                cached_results.append((x1, y1, x2, y2, label, color))

        # Rysuj buforowane wyniki na bieżącej klatce (bardzo szybko)
        out = frame.copy()
        for x1, y1, x2, y2, label, color in cached_results:
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                out,
                label,
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        cv2.putText(
            out,
            f"co {process_every} klatki | q/ESC = wyjscie",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow("FaceSearch - Live Preview", out)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
