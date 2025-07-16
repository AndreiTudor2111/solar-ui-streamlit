#!/usr/bin/env python3
import os, sys, argparse, json, requests
import pandas as pd
from io import BytesIO
from datasets import Dataset
from huggingface_hub import HfApi, create_repo

AUTO_TRAIN_PROJECT = "autotrain-advanced"

def push_dataset_to_hub(df, repo_id, token):
    ds = Dataset.from_pandas(df)
    create_repo(repo_id, exist_ok=True, token=token, repo_type="dataset")
    ds.push_to_hub(repo_id, token=token)
    print(f"✅ Dataset: https://huggingface.co/datasets/{repo_id}")

def launch_autotrain_job(repo_id, token, hf_username, config):
    url = f"https://api-inference.huggingface.co/auto-train/v2/projects/{hf_username}/{AUTO_TRAIN_PROJECT}/jobs"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = {
        "dataset": repo_id,
        "task": "tabular-classification",
        "column_mapping": {
            "target": "will_charge",
            "continuous_features": config.get("feature_columns", [])
        },
        "optim_args": {
            "num_train_epochs": config["epochs"],
            "batch_size": config["batch_size"]
        },
        "model_config": {
            "model_size": "small",
            "learning_rate": config["learning_rate"]
        }
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    print(f"📡 Job response: {resp.status_code} - {resp.json()}")
    job = resp.json()
    job_id = job.get("job_id") or job.get("id")
    print(f"✅ AutoTrain job: https://huggingface.co/autotrain/{hf_username}/{AUTO_TRAIN_PROJECT}/jobs/{job_id}")
    return job_id

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--user_id", required=True)
    p.add_argument("--run_id",  required=True)
    p.add_argument("--project", default=AUTO_TRAIN_PROJECT)
    args = p.parse_args()

    # 1) HF token + username
    token = open("HF_write_token.txt").read().strip()
    api = HfApi(token=token)
    who = api.whoami()
    hf_user = who.get("name") or who.get("user", {}).get("name")
    print(f"✅ HF user: {hf_user}")

    # 2) Încarcă configul de training din JSON
    config_path = os.path.join(os.getcwd(), "train_config.json")
    if not os.path.exists(config_path):
        print("❌ Nu există train_config.json! Anulez execuția.")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = json.load(f)

    # 3) Încarcă dataset local
    dataset_path = os.path.join("ui", "users", args.user_id, "Stan.csv")
    if not os.path.exists(dataset_path):
        print(f"❌ Datasetul nu există: {dataset_path}")
        sys.exit(1)

    df = pd.read_csv(dataset_path, parse_dates=["time"])
    features_path = os.path.join("ui", "users", args.user_id, "features.json")
    if not os.path.exists(features_path):
        print(f"❌ Nu există features.json pentru utilizator: {features_path}")
        sys.exit(1)
    
    with open(features_path, "r") as f:
        selected_features = json.load(f)
    config["feature_columns"] = selected_features


    # 4) Urcă datasetul pe Hub
    safe = args.user_id.replace("@", "-").replace(".", "-")
    repo = f"{hf_user}/{safe}-{args.run_id}-dataset"
    push_dataset_to_hub(df, repo, token)

    # 5) Lansează AutoTrain
    launch_autotrain_job(repo, token, hf_user, config)
    os.remove(config_path)

if __name__ == "__main__":
    main()


# #!/usr/bin/env python3
# import os, sys, argparse, json, requests
# import pandas as pd
# from io import BytesIO
# from datasets import Dataset
# from huggingface_hub import HfApi, create_repo

# # Numai numele proiectului AutoTrain de pe HF
# AUTO_TRAIN_PROJECT = "autotrain-advanced"

# def push_dataset_to_hub(df, repo_id, token):
#     ds = Dataset.from_pandas(df)
#     create_repo(repo_id, exist_ok=True, token=token, repo_type="dataset")
#     ds.push_to_hub(repo_id, token=token)
#     print(f"✅ Dataset: https://huggingface.co/datasets/{repo_id}")

# def launch_autotrain_job(repo_id, token, hf_username, feature_cols):
#     url = f"https://api-inference.huggingface.co/auto-train/v2/projects/{hf_username}/{AUTO_TRAIN_PROJECT}/jobs"
#     headers = {
#         "Authorization": f"Bearer {token}",
#         "Content-Type": "application/json"
#     }
#     payload = {
#         "dataset": repo_id,
#         "task": "tabular-classification",
#         "column_mapping": {
#             "target": "will_charge",
#             "continuous_features": feature_cols
#         },
#         "optim_args": {"num_train_epochs": 50, "batch_size": 64},
#         "model_config": {"model_size": "small"}
#     }
#     resp = requests.post(url, headers=headers, json=payload, timeout=60)
#     resp.raise_for_status()
#     job = resp.json()
#     job_id = job.get("job_id") or job.get("id")
#     print(f"✅ AutoTrain job: https://huggingface.co/autotrain/{hf_username}/{AUTO_TRAIN_PROJECT}/jobs/{job_id}")
#     return job_id

# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument("--user_id", required=True)
#     p.add_argument("--run_id",  required=True)
#     p.add_argument("--project", default="dizertatie-43ab8")
#     args = p.parse_args()

#     # 1) HF token + username
#     token = open("HF_write_token.txt").read().strip()
#     api = HfApi(token=token)
#     who = api.whoami()
#     hf_user = who.get("name") or who.get("user",{}).get("name")
#     print(f"✅ HF user: {hf_user}")

#     # 2) Încarcă Stan.csv din Firebase Storage (la fel cum făceai)
#     #    aici să ai deja df = pd.read_csv(...)
#     #    pentru exemplu, îl încărcăm local:
#     df = pd.read_csv(f"datasets/{args.user_id}/{args.run_id}/Stan.csv", parse_dates=["time"])

#     # 3) Urcă dataset pe Hub
#     safe = args.user_id.replace("@","-").replace(".","-")
#     repo = f"{hf_user}/{safe}-{args.run_id}-dataset"
#     push_dataset_to_hub(df, repo, token)

#     # 4) Lansează AutoTrain
#     feat_cols = [c for c in df.columns if c not in ["time","will_charge"]]
#     launch_autotrain_job(repo, token, hf_user, feat_cols)

# if __name__ == "__main__":
#     main()
