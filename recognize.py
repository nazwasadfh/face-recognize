import sys
import cv2
from insightface.app import FaceAnalysis
from db import recognize


def recognize_from_file(image_path: str, tolerance: float = 0.4):
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

    out = img.copy()
    for i, face in enumerate(faces, 1):
        x1, y1, x2, y2 = face.bbox[:4].astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(out.shape[1], x2), min(out.shape[0], y2)

        result = recognize(face.normed_embedding, tolerance=tolerance)
        if result:
            person, sim = result
            info = person["info"]
            label = f"{i}. {info['name']} {info['surname']} ({sim:.2f})"
            color = (0, 255, 0)
            print(f"✅ {label} | ID: {person['id']}")
        else:
            label = f"{i}. Nieznana osoba"
            color = (0, 0, 255)
            print(f"❌ {label}")

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, label, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imwrite("result.jpg", out)
    print("\nWynik z ramkami zapisano do result.jpg")
    cv2.imshow("Rozpoznawanie twarzy", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    tol = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
    recognize_from_file(sys.argv[1], tol)
