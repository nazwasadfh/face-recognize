# FaceSearch — lokalna baza rozpoznawania twarzy (CLI)

Wszystko działa na Twoim komputerze. Modele insightface pobierane są raz przy pierwszym uruchomieniu.

## Struktura

```
persons/<hash>/
├── fingerprint.npy   # wektor cech twarzy (512 wymiarów, ArcFace)
├── photo.png         # przycięte zdjęcie twarzy
└── info.json         # {"name": ..., "surname": ..., "secondname": ...}
```

## Instalacja

```bash
pip install -r requirements.txt
```

## Rozpoznawanie i dodawanie osób

```bash
python recognize.py zdjecie_testowe.jpg [tolerancja=0.4]
```

- Program wykrywa twarze i porównuje z bazą.
- Znane osoby są rozpoznawane po imieniu i nazwisku.
- Nieznane twarze są oznaczone jako `unknown #0`, `unknown #1` itd.
- Po zakończeniu analizy program pyta w konsoli `data for #N:` dla każdej nieznanej osoby.
  - Wciśnij **Enter** — osoba pozostaje nieznana i nie jest zapisywana.
  - Wpisz `imie nazwisko [drugieimie]` — osoba zostaje dodana do bazy (folder `persons/<hash>/` z `fingerprint.npy`, `photo.png` i `info.json`).
- Wynik z ramkami zapisano do `result.jpg`.

## Podgląd na żywo z kamery

```bash
python live.py [tolerancja=0.4] [kamera=0] [co_ile_klatek=2]
```

- Zielona ramka = rozpoznana osoba, czerwona = nieznana.
- Wyjście: `q` lub `ESC`.

## Przykład

```bash
# rozpoznaj i ewentualnie dodaj nieznane osoby
python recognize.py test.jpg 0.4

# podgląd na żywo
python live.py 0.4 0 2
```

## Uwagi prawne/etyczne

Dane biometryczne (odciski twarzy) to dane szczególnej kategorii (np. RODO w UE). Nie dodawaj osób do bazy bez ich zgody.
