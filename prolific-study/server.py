from flask import Flask, request, jsonify, send_from_directory
import json
import uuid
import os

app = Flask(__name__, static_folder="static", static_url_path="")

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

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

