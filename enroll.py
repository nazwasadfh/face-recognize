import sys
import cv2
from insightface.app import FaceAnalysis
from db import save_person


def enroll_from_file(image_path: str, name: str, surname: str, secondname: str):
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

    # bierzemy pierwszą wykrytą twarz
    face = faces[0]
    x1, y1, x2, y2 = face.bbox[:4].astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

    face_crop = img[y1:y2, x1:x2]

    person_id = save_person(
        name=name,
        surname=surname,
        secondname=secondname,
        embedding=face.embedding,
        normed_embedding=face.normed_embedding,
        face_image_bgr=face_crop,
    )

    print(f"✅ Zapisano osobę: {person_id}")
    print(f"   Folder: persons/{person_id}/")


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Użycie: python enroll.py <obraz.jpg> <imie> <nazwisko> <drugie_imie>")
        sys.exit(1)
    enroll_from_file(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
