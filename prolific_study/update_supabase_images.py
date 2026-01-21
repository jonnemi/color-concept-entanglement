from pathlib import Path
import os
import time
from supabase import create_client
from httpx import RemoteProtocolError, ReadTimeout, WriteError
from dotenv import load_dotenv

# ------------------
# CONFIG
# ------------------
load_dotenv()

SUPABASE_URL = "https://utwhgfveotpusdjopcnl.supabase.co/"
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE"]
BUCKET_NAME = "prolific_images"

LOCAL_IMAGE_ROOT = Path(
    "/mnt/lustre/work/eickhoff/esx061/color-concept-entanglement/data"
)

# Objects to update
IMAGE_PRIORS_OBJECTS = [
    "Rottweiler",
    "Sealyham terrier",
    "curly-coated retriever",
    "dalmatian",
    "espresso maker",
    "flat-coated retriever",
    "flute",
    "radio",
    "screw",
    "strainer",
    "typewriter",
    "van",
    "waffle iron",
]

COUNTERFACT_OBJECTS = [
    "Band Aid",
    "French horn",
    "Pomeranian",
    "car wheel",
    "faucet",
    "fridge",
    "limousine",
    "padlock",
    "plastic bag",
    "saxophone",
    "thimble",
    "truck",
    "trumpet",
    "wagon",
]

# Paths
IMAGE_PRIORS_ROOT = LOCAL_IMAGE_ROOT / "color_images/gpt-4o/image_priors"
COUNTERFACT_ROOT = LOCAL_IMAGE_ROOT / "color_images/gpt-4o/counterfact"

MAX_RETRIES = 6
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
# HELPERS
# ------------------
def normalize_object_name(name: str) -> str:
    return name.lower().replace(" ", "_")

def collect_object_folders(root: Path, object_names):
    folders = []
    wanted = {normalize_object_name(o) for o in object_names}

    for d in root.iterdir():
        if not d.is_dir():
            continue

        folder_prefix = d.name.lower()
        for obj in wanted:
            if folder_prefix.startswith(obj):
                folders.append(d)
                break

    return folders

# ------------------
# COLLECT FILES
# ------------------
folders_to_update = []

folders_to_update += collect_object_folders(
    IMAGE_PRIORS_ROOT, IMAGE_PRIORS_OBJECTS
)
folders_to_update += collect_object_folders(
    COUNTERFACT_ROOT, COUNTERFACT_OBJECTS
)

if not folders_to_update:
    raise RuntimeError("No folders found to update — check object names.")

image_paths = []

for folder in folders_to_update:
    for img in sorted(folder.glob("*.png")):
        rel_path = img.relative_to(LOCAL_IMAGE_ROOT)
        image_paths.append(rel_path)

print(f"Overwriting {len(image_paths)} images")
print("Folders:")
for f in folders_to_update:
    print(" -", f)

# ------------------
# UPLOAD (OVERWRITE)
# ------------------
for i, rel_path in enumerate(image_paths, 1):
    local_path = LOCAL_IMAGE_ROOT / rel_path

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with open(local_path, "rb") as f:
                supabase.storage.from_(BUCKET_NAME).update(
                    path=str(rel_path),
                    file=f,
                    file_options={"content-type": "image/png"},
                )

            print(f"[{i}/{len(image_paths)}] Updated {rel_path}")
            break

        except WriteError:
            print(f"WriteError ({attempt}/{MAX_RETRIES}) → resetting client")
            supabase = make_supabase()
            time.sleep(WRITE_ERROR_SLEEP)

        except (RemoteProtocolError, ReadTimeout):
            print(f"Network error ({attempt}/{MAX_RETRIES}) on {rel_path}")
            time.sleep(RETRY_SLEEP)

        except Exception:
            raise

    time.sleep(SLEEP_BETWEEN_UPLOADS)

print("Done.")
