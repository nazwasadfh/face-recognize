#!/usr/bin/env python3
"""
Importuje dane z plików JSON w folderze input/ do persons/<hash>/info.json.

Dla każdego pliku input/*.json:
1. Szuka imienia i nazwiska (w różnych możliwych kluczach).
2. Sprawdza czy osoba o takim imieniu i nazwisku istnieje w persons/*/info.json.
3. Jeśli tak -> dopisuje dane z input do info.json osoby (pole "imported_from_input").
4. Jeśli nie -> wypisuje "nie znaleziono $nazwisko $imie" i zostawia plik w input/.
5. Jeśli tak -> usuwa plik z input/.
"""

import json
import sys
from pathlib import Path

INPUT_DIR = Path("input")
PERSONS_DIR = Path("persons")

# Możliwe klucze imienia i nazwiska w plikach JSON z input/
NAME_KEYS = ("imie", "imię", "name", "first_name", "firstname", "imya")
SURNAME_KEYS = ("nazwisko", "surname", "last_name", "lastname", "familia")


def find_name_and_surname(data: dict):
    """Zwraca (imie, nazwisko) lub (None, None) jeśli nie znaleziono."""
    imie = None
    nazwisko = None

    for key in NAME_KEYS:
        if key in data and data[key]:
            imie = str(data[key]).strip()
            break

    for key in SURNAME_KEYS:
        if key in data and data[key]:
            nazwisko = str(data[key]).strip()
            break

    return imie, nazwisko


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
    """Dopisuje dane z input do info.json osoby."""
    with open(info_path, encoding="utf-8") as f:
        info = json.load(f)

    if "imported_from_input" not in info:
        info["imported_from_input"] = []

    # Dodajemy dane z input jako nowy wpis
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
            print(f"❌ {json_file.name}: nie można wczytać JSON ({e}) — pomijam, zostawiam plik.")
            continue

        imie, nazwisko = find_name_and_surname(data)

        if not imie or not nazwisko:
            print(f"⚠️  {json_file.name}: nie znaleziono imienia i nazwiska w pliku — zostawiam.")
            continue

        person_dir, info = find_person_by_name(imie, nazwisko)

        if person_dir is None:
            print(f"❌ nie znaleziono {nazwisko} {imie} — zostawiam plik {json_file.name}")
            continue

        # Znaleziono osobę -> dopisz dane do info.json i usuń plik z input/
        info_path = person_dir / "info.json"
        import_data_to_info(info_path, data)
        json_file.unlink()
        print(f"✅ {nazwisko} {imie}: dopisano dane do {person_dir.name}/info.json, usunięto {json_file.name}")

    print("\nGotowe.")


if __name__ == "__main__":
    main()
