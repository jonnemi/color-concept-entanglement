import csv
import requests
from pathlib import Path
from collections import defaultdict
import time

# ------------------
# CONFIG
# ------------------

SUPABASE_PUBLIC_BASE = (
    "https://utwhgfveotpusdjopcnl.supabase.co"
    "/storage/v1/object/public/prolific_images/"
)

LOCAL_IMAGE_ROOT = Path(
    "/mnt/lustre/work/eickhoff/esx061/color-concept-entanglement/data"
)

LOCAL_TABLE_ROOT = Path(
    "/mnt/lustre/work/eickhoff/esx061/color-concept-entanglement/data/prolific_stimuli"
)

CSV_FILES = [
    "stimulus_table_image_priors.csv",
    "stimulus_table_shapes.csv",
    "stimulus_table_counterfact.csv",
]

REQUEST_TIMEOUT = 10
SLEEP_BETWEEN_REQUESTS = 0.05

# ------------------
# COLLECT IMAGE PATHS
# ------------------

image_paths = set()
csv_sources = defaultdict(list)

for csv_file in CSV_FILES:
    path = LOCAL_TABLE_ROOT / csv_file
    if not path.exists():
        print(f" CSV not found: {csv_file}")
        continue

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = row["image_path"].strip()
            image_paths.add(img)
            csv_sources[img].append(csv_file)

print(f"\nFound {len(image_paths)} unique image paths\n")

# ------------------
# CHECK IMAGES
# ------------------

missing_local = []
http_errors = []
ok_images = []

for i, rel_path in enumerate(sorted(image_paths), 1):
    local_path = LOCAL_IMAGE_ROOT / rel_path
    public_url = SUPABASE_PUBLIC_BASE + rel_path

    print(f"[{i}/{len(image_paths)}] Checking {rel_path}")

    # --- local existence ---
    if not local_path.exists():
        missing_local.append(rel_path)
        print("   Missing locally")
        continue

    # --- public URL ---
    try:
        r = requests.get(public_url, timeout=REQUEST_TIMEOUT)

        if r.status_code != 200:
            http_errors.append((rel_path, r.status_code))
            print(f"   HTTP {r.status_code}")
        else:
            ct = r.headers.get("Content-Type", "")
            if not ct.startswith("image"):
                http_errors.append((rel_path, f"bad content-type: {ct}"))
                print(f"   Bad content-type: {ct}")
            else:
                ok_images.append(rel_path)
                print("   OK")

    except requests.RequestException as e:
        http_errors.append((rel_path, str(e)))
        print(f"   Request error: {e}")

    time.sleep(SLEEP_BETWEEN_REQUESTS)

# ------------------
# SUMMARY
# ------------------

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"Total images checked: {len(image_paths)}")
print(f"OK: {len(ok_images)}")
print(f"Missing locally: {len(missing_local)}")
print(f"HTTP / access errors: {len(http_errors)}")

if missing_local:
    print("\n MISSING LOCALLY:")
    for p in missing_local:
        print(f"  - {p} (referenced in {csv_sources[p]})")

if http_errors:
    print("\n PUBLIC ACCESS ERRORS:")
    for p, err in http_errors:
        print(f"  - {p}: {err} (from {csv_sources[p]})")

if not missing_local and not http_errors:
    print("\n All images are present and publicly accessible.")
    print("Likely cause of failures: transient network / CDN issues.")

print("=" * 60)
