# Multimodal Misinformation Evaluation Resources

This repository contains resources for evaluating multimodal misinformation detection models. It includes **two experiment directories**:

* `RedDot`
* `outcontext-misinfo-progress`

Each directory contains **evaluation-ready datasets, pretrained model checkpoints, and evaluation scripts** for reproducing the reported results.

---

# Repository Structure

```
relevant-evidence-detection/
│
├── checkpoints/
│   └── best model checkpoint (.pt)
│
├── data/
│   └── eval/
│       ├── DP/
│       ├── FIVEPILS/
│       ├── MMFAKEBENCH/
│       └── VERITE/
│
└── checkpoint-evaluation.py
├── model.py
├── utils.py


outcontext-misinfo-progress/
│
├── checkpoints/
│   └── best model checkpoint (.pt)
│
├── data/
│   └── eval/
│       ├── DP/
│       ├── FIVEPILS/
│       ├── MMFAKEBENCH/
│       └── VERITE/
│
└── checkpoint-evaluation.py
├── model.py
├── utils.py
```
---
## Example: Collecting Tweet Text and Image Data (X API)

The CRAVE dataset was collected by retrieving publicly available posts from the **X (formerly Twitter) API**, including tweet text and associated media.
Below is a simplified example demonstrating how tweet text and image data can be retrieved and stored using the **tweet ID as the filename**.

### Requirements

Install the Python client:

```bash
pip install tweepy requests
```

### Example Code

```python
import tweepy
import requests
import os
import json

# Replace with your credentials
BEARER_TOKEN = "YOUR_BEARER_TOKEN"

client = tweepy.Client(bearer_token=BEARER_TOKEN)

# Example tweet id
tweet_id = "1234567890123456789"

response = client.get_tweet(
    tweet_id,
    expansions=["attachments.media_keys"],
    media_fields=["url"],
    tweet_fields=["text", "created_at"]
)

tweet = response.data
media = {m.media_key: m for m in response.includes.get("media", [])}

# Create directory
os.makedirs("tweet_data", exist_ok=True)

tweet_record = {
    "tweet_id": tweet_id,
    "text": tweet.text,
    "created_at": str(tweet.created_at),
    "images": []
}

# Download images and name them using tweet id
if tweet.attachments:
    for i, key in enumerate(tweet.attachments["media_keys"]):
        image_url = media[key].url

        image_filename = f"{tweet_id}_{i}.jpg"
        image_path = f"tweet_data/{image_filename}"

        img_data = requests.get(image_url).content
        with open(image_path, "wb") as f:
            f.write(img_data)

        tweet_record["images"].append(image_filename)

# Save tweet text with tweet id filename
json_filename = f"tweet_data/{tweet_id}.json"
with open(json_filename, "w") as f:
    json.dump(tweet_record, f, indent=2)

print(f"Saved tweet metadata to {json_filename}")
```

### Output Structure

```
tweet_data/
│
├── 1234567890123456789.json
├── 1234567890123456789_0.jpg
├── 1234567890123456789_1.jpg
```

The **tweet ID is used as the identifier for both text and images**, making it easy to associate tweet metadata with its corresponding media files.



---

# Evaluation Datasets

Each experiment directory contains a `data/eval/` folder with **pre-computed feature files** for evaluation on the following datasets:

* **DP**

  * Dataset Explorer is available at: https://decade.ac.uk/data_annotation_dashboard/verify_data
  * Can be downloaded from: https://decade.ac.uk/crave_dataset/
  * Example download command:

    ```
    wget -r -np -nH --cut-dirs=1 https://decade.ac.uk/crave_dataset/
    ```

* **FIVEPILS**

* **MMFakeBench**

* **VERITE**


These files contain **pre-extracted multimodal features** (e.g., image embeddings and associated metadata) so that evaluation can be run directly **without recomputing features**.

---

# Model Checkpoints

The `checkpoints/` directory in each experiment folder contains the **best-performing trained model checkpoint** used for evaluation.

These checkpoints can be loaded directly by the evaluation script.

---

# Running Evaluation

To evaluate a model checkpoint on the datasets:

```bash
python checkpoint-evaluation.py
```

The script loads:

* the model checkpoint from `checkpoints/`
* the evaluation features from `data/eval/`

and reports the model’s performance on the available datasets.

---

# Notes

* All datasets in `data/eval/` are **evaluation feature files only**, not the original raw datasets.
* Feature files are provided in `.npy` format to allow **fast reproducible evaluation**.
* The provided checkpoints correspond to the **best validation performance** during training.

---
