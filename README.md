# FaceSearch - local face recognition database (CLI)

FaceSearch is a local, private face recognition system. Think of it as your own personal police database, but running entirely on your own computer - nothing is ever sent to the cloud.

You add people by photographing them. The program detects the face, crops it, and saves a unique "face fingerprint" along with the person's name and a cropped photo in a folder named with a unique hash.

Later, you can show it a new photo (or use your webcam live) and it will tell you who is in the picture by comparing faces with your local database. If it doesn't recognize someone, you can add them by typing their name.

Everything runs on your computer. InsightFace models are downloaded once on first run.

## Structure

```
persons/<hash>/
├── fingerprints/
│   ├── 0.npy   # first face embedding (ArcFace, 512 dims)
│   ├── 1.npy   # next embeddings (added on confirmation)
│   └── ...
├── photo.png   # cropped face photo
└── info.json   # {"name": ..., "surname": ..., "secondname": ..., "imported_from_input": [...]}
```

## Installation

```bash
pip install -r requirements.txt
```

`requirements.txt` contains `onnxruntime-gpu`, so the program tries to use GPU (NVIDIA CUDA) via the `CUDAExecutionProvider` by default. If the cuDNN library is missing on the system, the program automatically falls back to CPU with a warning.

### cuDNN (required for GPU)

`onnxruntime-gpu` needs the cuDNN library to run on CUDA. **cuDNN is not in `requirements.txt`** because it depends on your installed CUDA version. Install the matching package from PyPI:

- For CUDA 13.x:
  ```bash
  python -m pip install nvidia-cudnn-cu13
  ```
- For CUDA 12.x:
  ```bash
  python -m pip install nvidia-cudnn-cu12
  ```

Check your CUDA version (`nvidia-smi` shows "CUDA Version"), then install the matching package. After installation, restart the program — it will automatically detect `CUDAExecutionProvider` and use the GPU.

## Recognition and adding people

```bash
python recognize.py test_photo.jpg [tolerance=0.4] [--device cuda|cpu|auto]
```

- The program detects faces and compares them with the database.
- `--device` selects inference device: `cuda` (GPU, default), `cpu`, or `auto` (GPU if available). Falls back to CPU automatically if CUDA/cuDNN is unavailable.
- Known people are recognized by name and surname.
- Unknown faces are labeled `unknown #0`, `unknown #1`, etc. (number = face index on the image).
- After analysis, the program asks `data for #N:` for each unknown face.
  - Press **Enter** - the person remains unknown and is not saved.
  - Type `name surname [secondname]` - the person is added to the database (`persons/<hash>/` with `fingerprints/0.npy`, `photo.png`, and `info.json`).
- If the best match is ≥ 0.30, the program asks `Is #N Name Surname? (Y/n)` - Enter = Y.
  - If you answer **n**, the program asks `data for #N:` and checks if that person already exists in the database:
    - If yes → adds a new fingerprint to the existing person.
    - If no → creates a new person.
- If the similarity is ≥ 0.99, the program reports it as a duplicate (green box, "this is 100% the same person") and does not ask to add a fingerprint.
- Result with boxes is saved to `result.jpg` and opened with `eog` (if available).

## Live camera preview

```bash
python live.py <process_every> [camera=0] [tolerance=0.4] [--device cuda|cpu|auto]
```

- `<process_every>` **required** - process every N-th frame (e.g. `2` = every 2nd frame).
- Green box = recognized person, red box = unknown.
- Press `q` or ESC to exit.
- `--device` selects inference device: `cuda` (GPU, default), `cpu`, or `auto` (GPU if available). Falls back to CPU automatically if CUDA/cuDNN is unavailable.

## Importing data from input/

```bash
python import_input.py
```

- Reads all `input/*.json` files (each has `name: "First Last"`).
- Looks for a person with that name and surname in `persons/*/info.json`.
- If found → appends the JSON data to the person's `info.json` (under `imported_from_input`) and deletes the input file.
- If not found → prints `not found $surname $name` and leaves the file in `input/`.

## Searching people by name

```bash
python search.py <name> [--exact]
```

- Partial case-insensitive match by default (searches name, surname, secondname).
- `--exact` for exact name match (case-insensitive).
- Prints `person_id` and the full `info.json` content for each match.

## Legal / ethical notice

Biometric data (face embeddings) are special category data (e.g. GDPR in the EU). Do not add people to the database without their consent.
