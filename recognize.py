import sys
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from db import recognize, save_person


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
    unknown_faces = []  # lista: {"index": int, "embedding": np.ndarray, "face_crop": np.ndarray}
    unknown_counter = 0

    for i, face in enumerate(faces, 1):
        x1, y1, x2, y2 = face.bbox[:4].astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(out.shape[1], x2), min(out.shape[0], y2)

        result = recognize(face.normed_embedding, tolerance=tolerance)
        if result:
            person, sim = result
            info = person["info"]
            label = f"{info['name']} {info['surname']} ({sim:.2f})"
            color = (0, 255, 0)  # zielony – rozpoznano
            print(f"✅ {label} | ID: {person['id']}")
        else:
            label = f"unknown #{unknown_counter}"
            color = (0, 0, 255)  # czerwony – nieznana
            print(f"❌ {label}")

            face_crop = img[y1:y2, x1:x2]
            unknown_faces.append({
                "index": unknown_counter,
                "embedding": face.normed_embedding,
                "face_crop": face_crop,
            })
            unknown_counter += 1

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, label, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imwrite("result.jpg", out)
    print("\nWynik z ramkami zapisano do result.jpg")

    # Interaktywny zapis nieznanych osób
    if unknown_faces:
        print("\n--- Dane dla nieznanych osób ---")
        for uf in unknown_faces:
            user_input = input(f"data for #{uf['index']}: ").strip()
            if not user_input:
                print(f"   ⏭️  Pominięto #{uf['index']} (pozostaje unknown)")
                continue

            parts = user_input.split()
            name = parts[0] if len(parts) > 0 else ""
            surname = parts[1] if len(parts) > 1 else ""
            secondname = parts[2] if len(parts) > 2 else ""

            if not name and not surname:
                print(f"   ⏭️  Pominięto #{uf['index']} (brak imienia i nazwiska)")
                continue

            person_id = save_person(
                name=name,
                surname=surname,
                secondname=secondname,
                embedding=uf["embedding"],
                normed_embedding=uf["embedding"],
                face_image_bgr=uf["face_crop"],
            )
            print(f"   ✅ Zapisano #{uf['index']} jako: {person_id} ({name} {surname} {secondname})")
    else:
        print("Brak nieznanych osób do uzupełnienia.")

    cv2.imshow("Rozpoznawanie twarzy", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    tol = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
    recognize_from_file(sys.argv[1], tol)
