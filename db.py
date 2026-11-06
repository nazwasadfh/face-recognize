import json
import uuid
from pathlib import Path
import numpy as np
import cv2

PERSONS_DIR = Path("persons")
PERSONS_DIR.mkdir(parents=True, exist_ok=True)


def generate_person_id() -> str:
    """Unikalny hash osoby (UUID4 hex). Praktycznie niemożliwy do powtórzenia."""
    return uuid.uuid4().hex


def save_person(name: str, surname: str, secondname: str,
                embedding: np.ndarray, normed_embedding: np.ndarray,
                face_image_bgr: np.ndarray) -> str:
    """
    Zapisuje nową osobę w persons/<hash>/
    - fingerprint.npy → znormalizowany wektor cech (512 wymiarów)
    - photo.png → przycięte zdjęcie twarzy (BGR)
    - info.json → dane tekstowe
    """
    person_id = generate_person_id()
    person_dir = PERSONS_DIR / person_id
    person_dir.mkdir(exist_ok=True)

    np.save(person_dir / "fingerprint.npy", normed_embedding)
    cv2.imwrite(str(person_dir / "photo.png"), face_image_bgr)

    info = {"name": name, "surname": surname, "secondname": secondname}
    with open(person_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    return person_id


def load_all_persons():
    """Wczytuje wszystkie osoby z bazy jako listę słowników."""
    persons = []
    for person_dir in PERSONS_DIR.iterdir():
        if not person_dir.is_dir():
            continue
        fp = person_dir / "fingerprint.npy"
        info_path = person_dir / "info.json"
        if not fp.exists() or not info_path.exists():
            continue
        emb = np.load(fp)
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
        persons.append({"id": person_dir.name, "info": info, "embedding": emb})
    return persons


def recognize(normed_embedding: np.ndarray, tolerance: float = 0.4):
    """
    Porównuje podany wektor (znormalizowany) z bazą metodą cosinusową.
    Zwraca (person_dict, similarity) lub None jeśli brak dopasowania.
    """
    best = None
    best_sim = -1.0
    for person in load_all_persons():
        sim = float(np.dot(person["embedding"], normed_embedding))
        if sim >= tolerance and sim > best_sim:
            best_sim = sim
            best = person
    if best is None:
        return None
    return best, best_sim
