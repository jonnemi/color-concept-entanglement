import csv
from pathlib import Path
import os
from dotenv import load_dotenv
import time
from supabase import create_client
from httpx import RemoteProtocolError, ReadTimeout, WriteError

# ------------------
# CONFIG
# ------------------
load_dotenv()

SUPABASE_URL = "https://utwhgfveotpusdjopcnl.supabase.co"
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE"]
BUCKET_NAME = "prolific_images"

LOCAL_IMAGE_ROOT = Path("/mnt/lustre/work/eickhoff/esx061/color-concept-entanglement/data")
LOCAL_TABLE_ROOT = Path("/mnt/lustre/work/eickhoff/esx061/color-concept-entanglement/data/prolific_stimuli")
CSV_FILES = [
    # "stimulus_table_counterfact.csv",
    "stimulus_table_image_priors.csv",
    "stimulus_table_shapes.csv",
]

MAX_RETRIES = 6
RESET_EVERY = 150
SLEEP_BETWEEN_UPLOADS = 0.15
RETRY_SLEEP = 3.0
WRITE_ERROR_SLEEP = 5.0

# ------------------
# SUPABASE CLIENT
# ------------------
def make_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = make_supabase()

# ------------------
# COLLECT IMAGE PATHS
# ------------------
image_paths = set()

for csv_file in CSV_FILES:
    with open(LOCAL_TABLE_ROOT / csv_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_paths.add(row["image_path"])

print(f"Found {len(image_paths)} unique images")

# ------------------
# UPLOAD
# ------------------
for i, rel_path in enumerate(sorted(image_paths), 1):
    local_path = LOCAL_IMAGE_ROOT / rel_path

    if not local_path.exists():
        print(f"Missing: {local_path}")
        continue

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with open(local_path, "rb") as f:
                supabase.storage.from_(BUCKET_NAME).upload(
                    path=rel_path,
                    file=f,
                    file_options={"content-type": "image/png"},
                )
            print(f"Uploaded {rel_path}")
            break

        except WriteError:
            # TLS connection is dead → must recreate client
            print(f"WriteError ({attempt}/{MAX_RETRIES}) → resetting client")
            supabase = make_supabase()
            time.sleep(WRITE_ERROR_SLEEP)

        except (RemoteProtocolError, ReadTimeout):
            print(f"Network error ({attempt}/{MAX_RETRIES}) on {rel_path}")
            time.sleep(RETRY_SLEEP)

        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"Exists, skipping {rel_path}")
                break
            else:
                raise

    if i % RESET_EVERY == 0:
        print("Periodic Supabase client reset")
        supabase = make_supabase()

    time.sleep(SLEEP_BETWEEN_UPLOADS)
