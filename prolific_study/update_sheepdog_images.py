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

FOLDERS_TO_UPDATE = [
    LOCAL_IMAGE_ROOT / "color_images/gpt-4o/image_priors/Old_English_sheepdog_4_51cd66d5_resized_grey",
    LOCAL_IMAGE_ROOT / "color_images/gpt-4o/counterfact/Old_English_sheepdog_4_51cd66d5_resized_yellow",
]

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
# COLLECT FILES
# ------------------
image_paths = []

for folder in FOLDERS_TO_UPDATE:
    if not folder.exists():
        raise RuntimeError(f"Folder not found: {folder}")

    for img in sorted(folder.glob("*.png")):
        rel_path = img.relative_to(LOCAL_IMAGE_ROOT)
        image_paths.append(rel_path)

print(f"Overwriting {len(image_paths)} images")

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
                    file_options={"content-type": "image/png"}
                )

            print(f"Updated {rel_path}")
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
