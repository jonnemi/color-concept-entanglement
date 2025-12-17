from flask import Flask, request, jsonify, send_from_directory
import json
import uuid
import os
import pandas as pd
import hashlib
from sample_experiment import sample_experiment_1


app = Flask(__name__, static_folder="static", static_url_path="")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STIMULUS_TABLE = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "..",
        "data",
        "prolific_stimuli",
        "stimulus_table.csv"
    )
)


def seed_from_prolific_id(pid: str) -> int:
    return int(
        hashlib.sha256(pid.encode()).hexdigest(),
        16
    ) % (10**8)

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/get_stimuli")
def get_stimuli():
    prolific_id = request.args.get("PROLIFIC_PID")
    if prolific_id is None:
        return jsonify({"error": "Missing PROLIFIC_PID"}), 400

    seed = seed_from_prolific_id(prolific_id)

    sampled = sample_experiment_1(
        STIMULUS_TABLE,
        seed=seed
    )

    # Convert to JSON-safe format
    records = sampled.to_dict(orient="records")

    return jsonify({
        "seed": seed,
        "stimuli": records
    })

@app.route("/finish.html")
def finish():
    return send_from_directory("static", "finish.html")


@app.route("/save-json", methods=["POST"])
def save_json():
    try:
        payload = request.get_json()
        exp = payload.get("experiment_type", "unknown")
        data = payload.get("data", {})

        session_id = str(uuid.uuid4())
        save_dir = f"data/exp{exp}"
        os.makedirs(save_dir, exist_ok=True)

        with open(f"{save_dir}/session-{session_id}.json", "w") as f:
            json.dump(data, f, indent=2)

        return jsonify({"message": "Saved"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

