from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path
import json
import hashlib
import uuid
import os

# ---------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------

app = Flask(__name__, static_folder="static", static_url_path="")

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------
# Profile loading
# ---------------------------------------------------------------------

PROFILE_DIR = BASE_DIR / "static" /"img" / "dataset" / "prolific_stimuli" / "profiles"
PROFILE_FILES = sorted(PROFILE_DIR.glob("profile_*.json"))

N_PROFILES = len(PROFILE_FILES)
assert N_PROFILES == 74, f"Expected 74 profiles, found {N_PROFILES}"

# ---------------------------------------------------------------------
# Deterministic profile assignment
# ---------------------------------------------------------------------

def assign_profile_index(prolific_pid: str) -> int:
    """
    Deterministically assign a participant to one of the precomputed profiles.
    """
    h = hashlib.sha256(prolific_pid.encode()).hexdigest()
    return int(h, 16) % N_PROFILES

# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/get_profile")
def get_profile():
    """
    Return the assigned survey profile for a given Prolific participant.
    """
    prolific_pid = request.args.get("PROLIFIC_PID")

    if not prolific_pid:
        return jsonify({"error": "Missing PROLIFIC_PID"}), 400

    profile_idx = assign_profile_index(prolific_pid)
    profile_path = PROFILE_FILES[profile_idx]

    with open(profile_path, "r") as f:
        profile = json.load(f)

    return jsonify({
        "profile_id": profile_path.stem,
        "profile_index": profile_idx,
        "questions": profile["questions"],
    })


@app.route("/save_results", methods=["POST"])
def save_results():
    """
    Save participant responses.
    """
    payload = request.get_json()

    prolific_pid = payload.get("PROLIFIC_PID", "UNKNOWN")
    profile_id = payload.get("profile_id", "UNKNOWN")
    data = payload.get("data", [])

    out_dir = BASE_DIR / "results"
    out_dir.mkdir(exist_ok=True)

    session_id = uuid.uuid4().hex
    out_path = out_dir / f"{prolific_pid}_{profile_id}_{session_id}.json"

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    return jsonify({"status": "ok"}), 200


@app.route("/finish.html")
def finish():
    return send_from_directory("static", "finish.html")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
