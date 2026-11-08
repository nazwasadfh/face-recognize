#!/usr/bin/env python3
"""
Search people by name.

Usage:
    python search.py <name>           # partial match (case-insensitive)
    python search.py <name> --exact   # exact match (case-insensitive)

Prints person_id and full info.json content for found people.
"""

import json
import sys
from pathlib import Path

PERSONS_DIR = Path("persons")


def search_by_name(query: str, exact: bool = False):
    """Search for people by name in persons/*/info.json."""
    if not PERSONS_DIR.exists():
        print("Folder persons/ does not exist.")
        return []

    query_lower = query.strip().lower()
    results = []

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

        name = str(info.get("name", "")).strip().lower()
        surname = str(info.get("surname", "")).strip().lower()
        secondname = str(info.get("secondname", "")).strip().lower()

        match = False
        if exact:
            if name == query_lower:
                match = True
        else:
            if query_lower in name or query_lower in surname or query_lower in secondname:
                match = True

        if match:
            results.append({
                "person_id": person_dir.name,
                "info": info,
            })

    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python search.py <name> [--exact]")
        print("  <name>    - name to search for (partial match)")
        print("  --exact   - exact name match")
        sys.exit(1)

    query = sys.argv[1]
    exact = "--exact" in sys.argv

    results = search_by_name(query, exact=exact)

    if not results:
        print(f"No person found with name containing '{query}'.")
        return

    print(f"Found {len(results)} person(s):\n")
    for r in results:
        info = r["info"]
        print(f"ID: {r['person_id']}")
        print(json.dumps(info, ensure_ascii=False, indent=2))
        print("-" * 50)


if __name__ == "__main__":
    main()
