#!/usr/bin/env python3
"""
Importuje dane z plików JSON w folderze input/ do persons/<hash>/info.json.

Struktura plików input/*.json (identyczne klucze we wszystkich):
{
  "id": 10,
  "name": "Daniel Słyk",        <- imię i nazwisko razem
  "full_name": "Daniel Słyk",
  "alias": "Słoik",
  "birthday": null,
  "address": null,
  "photo": null,
  "phones": [{"number": "+48 ..."}],
  "emails": [],
  "custom": {...}
}

Dla każdego pliku:
1. Rozbija 'name' na imię (pierwsze słowo) i nazwisko (reszta).
2. Szuka osoby o takim imieniu i nazwisku w persons/*/info.json.
3. Jeśli znaleziono -> dopisuje dane z input do info.json (pole "imported_from_input") i usuwa plik z input/.
4. Jeśli nie znaleziono -> wypisuje "nie znaleziono $nazwisko $imie" i zostawia plik.
"""

import json
import sys
from pathlib import Path

INPUT_DIR = Path("input")
PERSONS_DIR = Path("persons")


def split_name(full_name: str):
    """Rozbija 'Imię Nazwisko' na (imie, nazwisko). Pierwsze słowo = imię, reszta = nazwisko."""
    parts = full_name.strip().split()
    if len(parts) < 2:
        return parts[0] if parts else "", ""
    return parts[0], " ".join(parts[1:])


def find_person_by_name(name: str, surname: str):
    """Szuka osoby po imieniu i nazwisku w persons/*/info.json. Zwraca (person_dir, info) lub (None, None)."""
    if not PERSONS_DIR.exists():
        return None, None
    for person_dir in PERSONS_DIR.iterdir():
        if not person_dir.is_dir():
            continue
        info_path = person_dir / "info.json"
        if not info_path.exists():
            continue
        try:
            with open(info_path, encoding="utf-8") as f:
                info = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        info_name = str(info.get("name", "")).strip().lower()
        info_surname = str(info.get("surname", "")).strip().lower()
        if info_name == name.strip().lower() and info_surname == surname.strip().lower():
            return person_dir, info
    return None, None


def import_data_to_info(info_path: Path, input_data: dict):
    """Dopisuje dane z input do info.json osoby (pod kluczem 'imported_from_input' jako lista)."""
    with open(info_path, encoding="utf-8") as f:
        info = json.load(f)

    if "imported_from_input" not in info:
        info["imported_from_input"] = []

    info["imported_from_input"].append(input_data)

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


def main():
    if not INPUT_DIR.exists():
        print(f"Folder {INPUT_DIR}/ nie istnieje — nic do importu.")
        return

    json_files = sorted(INPUT_DIR.glob("*.json"))
    if not json_files:
        print(f"Brak plików JSON w {INPUT_DIR}/.")
        return

    print(f"Znaleziono {len(json_files)} plik(ów) JSON w {INPUT_DIR}/\n")

    for json_file in json_files:
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"❌ {json_file.name}: nie można wczytać JSON ({e}) — zostawiam plik.")
            continue

        full_name = data.get("name") or data.get("full_name") or ""
        if not full_name:
            print(f"⚠️  {json_file.name}: brak pola 'name' — zostawiam plik.")
            continue

        imie, nazwisko = split_name(full_name)

        person_dir, info = find_person_by_name(imie, nazwisko)

        if person_dir is None:
            print(f"❌ nie znaleziono {nazwisko} {imie} — zostawiam plik {json_file.name}")
            continue

        # Znaleziono -> dopisz dane do info.json i usuń plik
        info_path = person_dir / "info.json"
        import_data_to_info(info_path, data)
        json_file.unlink()
        print(f"✅ {nazwisko} {imie}: dopisano dane do {person_dir.name}/info.json, usunięto {json_file.name}")

    print("\nGotowe.")


if __name__ == "__main__":
    main()
