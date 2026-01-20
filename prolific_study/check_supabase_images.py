import csv
import requests
from pathlib import Path
from PIL import Image
from io import BytesIO
from tqdm import tqdm

# ------------------
# CONFIG
# ------------------

SUPABASE_IMAGE_BASE = (
    "https://utwhgfveotpusdjopcnl.supabase.co"
    "/storage/v1/object/public/prolific_images/"
)

CSV_DIR = Path("data/prolific_stimuli")
CSV_FILES = [
    "stimulus_table_image_priors.csv",
    "stimulus_table_counterfact.csv",
    "stimulus_table_shapes.csv",
]

TIMEOUT = 10  # seconds

# ------------------
# COLLECT IMAGE PATHS
# ------------------

image_paths = set()

for csv_file in CSV_FILES:
    with open(CSV_DIR / csv_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_paths.add(row["image_path"])

print(f"Checking {len(image_paths)} images...\n")

# ------------------
# CHECK IMAGES
# ------------------

errors = []

for path in tqdm(sorted(image_paths)):
    url = SUPABASE_IMAGE_BASE + path

    try:
        r = requests.get(url, timeout=TIMEOUT)
        if r.status_code != 200:
            errors.append((path, f"HTTP {r.status_code}"))
            continue

        if len(r.content) < 100:
            errors.append((path, "File too small"))
            continue

        # Try decoding image
        img = Image.open(BytesIO(r.content))
        img.verify()  # checks integrity

    except Exception as e:
        errors.append((path, str(e)))

# ------------------
# REPORT
# ------------------

print("\n--- RESULT ---")
if not errors:
    print("All images loaded and decoded successfully.")
else:
    print(f"{len(errors)} problematic images:\n")
    for p, err in errors:
        print(f"{p} → {err}")

    # Optional: save for inspection
    with open("corrupt_images.txt", "w") as f:
        for p, err in errors:
            f.write(f"{p}\t{err}\n")

    print("\nSaved list to corrupt_images.txt")
