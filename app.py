import gradio as gr
import subprocess
import json

def train_from_payload(data):
    user_id = data["user_id"]
    run_id = data["run_id"]
    config = data["config"]

    # Creează temporar train_config.json
    with open("train_config.json", "w") as f:
        json.dump(config, f)

    # Rulează training
    cmd = ["python", "train.py", "--user_id", user_id, "--run_id", run_id]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout

iface = gr.Interface(fn=train_from_payload, inputs="json", outputs="text")
iface.launch()
