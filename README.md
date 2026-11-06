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

## Dodawanie osoby do bazy

```bash
python enroll.py zdjecie.jpg Jan Kowalski Nowy
```

Tworzy folder `persons/<uuid>/` z fingerprintiem, zdjęciem i danymi.

## Rozpoznawanie

```bash
python recognize.py zdjecie_testowe.jpg [tolerancja=0.4]
```

- Im niższa tolerancja, tym bardziej restrykcyjne dopasowanie.
- Wynik wypisywany jest w konsoli i zapisywany do `result.jpg`.

## Przykład

```bash
# rejestracja
python enroll.py jan.jpg Jan Kowalski Nowy

# rozpoznawanie
python recognize.py test.jpg 0.4
```

## Uwagi prawne/etyczne

Dane biometryczne (odciski twarzy) to dane szczególnej kategorii (np. RODO w UE). Nie dodawaj osób do bazy bez ich zgody.
