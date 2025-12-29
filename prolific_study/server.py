from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path
import json
import hashlib
import uuid
import os
from supabase import create_client, Client

# ---------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------

app = Flask(__name__, static_folder="static", static_url_path="")

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------
# Supabase setup
# ---------------------------------------------------------------------

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("Missing Supabase environment variables")

supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_SERVICE_KEY
)

# ---------------------------------------------------------------------
# Profile loading
# ---------------------------------------------------------------------

PROFILE_DIR = (
    BASE_DIR
    / "data"
    / "prolific_stimuli"
    / "profiles"
)


# Production profiles
PROFILE_FILES = sorted(
    p for p in PROFILE_DIR.glob("profile_*.json")
    if not p.name.startswith("debug")
)

N_PROFILES = len(PROFILE_FILES)
assert N_PROFILES == 74, f"Expected 74 profiles, found {N_PROFILES}"

# Debug profile (explicit, never part of assignment)
DEBUG_PROFILE_PATH = PROFILE_DIR / "debug_profile.json"
assert DEBUG_PROFILE_PATH.exists(), "debug_profile.json not found"

# ---------------------------------------------------------------------
# Deterministic profile assignment
# ---------------------------------------------------------------------

def assign_profile_index(prolific_pid: str) -> int:
    """
    Deterministically assign a participant to one of the production profiles.
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

    # --------------------------------------------------
    # DEBUG MODE
    # --------------------------------------------------
    if prolific_pid == "DEBUG":
        with open(DEBUG_PROFILE_PATH, "r") as f:
            profile = json.load(f)

        return jsonify({
            "profile_id": "debug_profile",
            "profile_index": -1,
            "questions": profile["questions"],
        })

    # --------------------------------------------------
    # PRODUCTION MODE
    # --------------------------------------------------
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
    payload = request.get_json()

    prolific_pid = payload.get("PROLIFIC_PID", "UNKNOWN")
    data = payload.get("data", [])

    # Pull metadata from jsPsych properties (first trial)
    meta = data[0] if data else {}

    row = {
        "prolific_pid": prolific_pid,
        "profile_id": meta.get("profile_id"),
        "profile_index": meta.get("profile_index"),
        "exit_reason": meta.get("exit_reason", "completed"),
        "experiment_start_time": meta.get("experiment_start_time"),
        "exit_time": meta.get("exit_time"),
        "data": data,
    }

    supabase.table("results").insert(row).execute()

    return jsonify({"status": "ok"}), 200


@app.route("/finish.html")
def finish():
    return send_from_directory("static", "finish.html")


@app.route("/decline.html")
def decline():
    return send_from_directory("static", "decline.html")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

