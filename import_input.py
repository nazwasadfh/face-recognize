#!/usr/bin/env python3
"""
Imports data from JSON files in folder input/ to persons/<hash>/info.json.

Structure of input/*.json files (identical keys in all):
{
  "id": 10,
  "name": "Daniel Słyk",        <- first and last name together
  "full_name": "Daniel Słyk",
  "alias": "Słoik",
  "birthday": null,
  "address": null,
  "photo": null,
  "phones": [{"number": "+48 ..."}],
  "emails": [],
  "custom": {...}
}

For each file:
1. Splits 'name' into first name (first word) and last name (rest).
2. Searches for a person with that first and last name in persons/*/info.json.
3. If found -> appends the input data to the person's info.json (field "imported_from_input") and deletes the file from input/.
4. If not found -> prints "not found $surname $name" and leaves the file in input/.
"""

import json
import sys
from pathlib import Path

INPUT_DIR = Path("input")
PERSONS_DIR = Path("persons")


def split_name(full_name: str):
    """Splits 'First Last' into (first_name, last_name). First word = first name, rest = last name."""
    parts = full_name.strip().split()
    if len(parts) < 2:
        return parts[0] if parts else "", ""
    return parts[0], " ".join(parts[1:])


def find_person_by_name(name: str, surname: str):
    """Searches for a person by first and last name in persons/*/info.json. Returns (person_dir, info) or (None, None)."""
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
    """Appends input data to the person's info.json (under 'imported_from_input' as a list)."""
    with open(info_path, encoding="utf-8") as f:
        info = json.load(f)

    if "imported_from_input" not in info:
        info["imported_from_input"] = []

    info["imported_from_input"].append(input_data)

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


def main():
    if not INPUT_DIR.exists():
        print(f"Folder {INPUT_DIR}/ does not exist — nothing to import.")
        return

    json_files = sorted(INPUT_DIR.glob("*.json"))
    if not json_files:
        print(f"No JSON files in {INPUT_DIR}/.")
        return

    print(f"Found {len(json_files)} JSON file(s) in {INPUT_DIR}/\n")

    for json_file in json_files:
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"❌ {json_file.name}: cannot load JSON ({e}) — keeping file.")
            continue

        full_name = data.get("name") or data.get("full_name") or ""
        if not full_name:
            print(f"⚠️  {json_file.name}: no 'name' field — keeping file.")
            continue

        first_name, last_name = split_name(full_name)

        person_dir, info = find_person_by_name(first_name, last_name)

        if person_dir is None:
            print(f"❌ not found {last_name} {first_name} — keeping file {json_file.name}")
            continue

        # Found -> append data to info.json and delete the file
        info_path = person_dir / "info.json"
        import_data_to_info(info_path, data)
        json_file.unlink()
        print(f"✅ {last_name} {first_name}: data appended to {person_dir.name}/info.json, deleted {json_file.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
