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


def _next_fingerprint_index(person_dir: Path) -> int:
    """Zwraca kolejny wolny indeks w folderze fingerprints/ danej osoby."""
    fp_dir = person_dir / "fingerprints"
    if not fp_dir.exists():
        return 0
    existing = []
    for p in fp_dir.glob("*.npy"):
        if p.stem.isdigit():
            existing.append(int(p.stem))
    return max(existing) + 1 if existing else 0


def save_person(name: str, surname: str, secondname: str,
                embedding: np.ndarray, face_image_bgr: np.ndarray) -> str:
    """
    Tworzy nową osobę w persons/<hash>/
    - fingerprints/0.npy → pierwszy wektor cech
    - photo.png → przycięte zdjęcie twarzy
    - info.json → dane tekstowe
    Zwraca person_id.
    """
    person_id = generate_person_id()
    person_dir = PERSONS_DIR / person_id
    fp_dir = person_dir / "fingerprints"
    fp_dir.mkdir(parents=True, exist_ok=True)

    idx = _next_fingerprint_index(person_dir)
    np.save(fp_dir / f"{idx}.npy", embedding)
    cv2.imwrite(str(person_dir / "photo.png"), face_image_bgr)

    info = {"name": name, "surname": surname, "secondname": secondname}
    with open(person_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    return person_id


def add_fingerprint(person_id: str, embedding: np.ndarray,
                    face_image_bgr: np.ndarray = None,
                    check_duplicates: bool = True,
                    duplicate_threshold: float = 0.99) -> int | None:
    """
    Dodaje kolejny fingerprint do istniejącej osoby.
    Zwraca indeks nowego pliku (np. 1, 2, 3...) lub None jeśli zablokowano jako duplikat.

    Parametry:
    - check_duplicates: jeśli True, sprawdza czy embedding nie jest już w bazie (>= threshold)
    - duplicate_threshold: próg acima którego uznajemy za duplikat (0.99 = prawie to samo zdjęcie)
    """
    if check_duplicates and is_duplicate_embedding(embedding, threshold=duplicate_threshold):
        return None  # zablokowano jako duplikat

    person_dir = PERSONS_DIR / person_id
    fp_dir = person_dir / "fingerprints"
    fp_dir.mkdir(parents=True, exist_ok=True)
    idx = _next_fingerprint_index(person_dir)
    np.save(fp_dir / f"{idx}.npy", embedding)
    if face_image_bgr is not None:
        cv2.imwrite(str(person_dir / "photo.png"), face_image_bgr)
    return idx


def load_all_persons():
    """Wczytuje wszystkie osoby z bazy, każda z listą wszystkich swoich embeddingsów."""
    persons = []
    for person_dir in PERSONS_DIR.iterdir():
        if not person_dir.is_dir():
            continue
        info_path = person_dir / "info.json"
        if not info_path.exists():
            continue
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)

        fp_dir = person_dir / "fingerprints"
        embeddings = []
        if fp_dir.exists():
            files = sorted(fp_dir.glob("*.npy"),
                           key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
            for fp_file in files:
                embeddings.append(np.load(fp_file))

        if not embeddings:
            continue

        persons.append({
            "id": person_dir.name,
            "info": info,
            "embeddings": embeddings,
        })
    return persons


def load_embeddings_matrix():
    """
    Zwraca (matrix, meta) do szybkiego batch porównania cosine similarity.

    matrix : np.ndarray (N, 512) – wszystkie fingerprints wszystkich osób
    meta   : lista słowników {person_id, info, fp_idx} odpowiadających wierszom
    """
    persons = load_all_persons()
    rows = []
    meta = []
    for p in persons:
        for i, emb in enumerate(p["embeddings"]):
            rows.append(emb)
            meta.append({
                "person_id": p["id"],
                "info": p["info"],
                "fp_idx": i,
            })
    if not rows:
        return np.empty((0, 512), dtype=np.float32), []
    return np.vstack(rows).astype(np.float32), meta


def is_duplicate_embedding(embedding: np.ndarray, threshold: float = 0.99) -> bool:
    """
    Sprawdza czy dany embedding jest duplikatem istniejącego fingerprintu w bazie.
    Zwraca True jeśli istnieje fingerprint z cosine similarity >= threshold.
    """
    matrix, _ = load_embeddings_matrix()
    if len(matrix) == 0:
        return False
    sims = matrix @ embedding  # wszystkie wektory są znormalizowane przez insightface
    best_sim = float(np.max(sims))
    return best_sim >= threshold


def clean_duplicates(threshold: float = 0.99) -> int:
    """
    Usuwa duplikaty fingerprintów z całej bazy.
    Dla każdego fingerprintu porównuje z wszystkimi poprzednio zachowanymi;
    jeśli similarity >= threshold, plik jest usuwany.
    Zwraca liczbę usuniętych plików.
    """
    persons = load_all_persons()
    all_fps = []  # (person_id, idx, fp_file_path, embedding)

    for p in persons:
        fp_dir = PERSONS_DIR / p["id"] / "fingerprints"
        if not fp_dir.exists():
            continue
        files = sorted(fp_dir.glob("*.npy"),
                       key=lambda pth: int(pth.stem) if pth.stem.isdigit() else 0)
        for fp_file in files:
            try:
                emb = np.load(fp_file)
            except Exception:
                continue
            idx = int(fp_file.stem) if fp_file.stem.isdigit() else 0
            all_fps.append((p["id"], idx, fp_file, emb))

    kept_indices = []  # indeksy w all_fps, które zostają
    removed = 0

    for i, (pid, idx, fp_file, emb) in enumerate(all_fps):
        is_dup = False
        for j in kept_indices:
            _, _, _, emb_j = all_fps[j]
            sim = float(np.dot(emb, emb_j))
            if sim >= threshold:
                is_dup = True
                break
        if is_dup:
            fp_file.unlink(missing_ok=True)
            removed += 1
        else:
            kept_indices.append(i)

    return removed
