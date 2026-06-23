# Multimodal Misinformation Evaluation Resources

**Dataset is hosted externally:**
at https://cvssp.org/data/crave

---

## Overview

This repository contains resources for evaluating multimodal misinformation detection models. It includes **two experiment setups**:

* `relevant-evidence-detection`
* `outcontext-misinfo-progress`

Each experiment includes:

* evaluation scripts
* pretrained model checkpoints
* configuration for reproducible evaluation

---

## Dataset Download

The full dataset is hosted externally by the Centre for Vision, Speech and Signal Processing (CVSSP):

👉 https://cvssp.org/data/crave

### Download

```bash
wget https://cvssp.org/data/crave/outcontext-misinfo-progress.tar.gz
wget https://cvssp.org/data/crave/relevant-evidence-detection.tar.gz
```

### Extract

```bash
tar -xzf outcontext-misinfo-progress.tar.gz
tar -xzf relevant-evidence-detection.tar.gz
```

---

## Repository Structure

⚠️ The dataset is **NOT included** in this repository and must be downloaded separately.

After downloading and extracting the dataset, your working directory should look like:

```
outcontext-misinfo-progress/
relevant-evidence-detection/
```

Each directory contains:

```
<experiment>/
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
├── checkpoint-evaluation.py
├── model.py
└── utils.py
```

---

## Running Evaluation

To evaluate a model checkpoint:

```bash
cd outcontext-misinfo-progress
python checkpoint-evaluation.py
```

or:

```bash
cd relevant-evidence-detection
python checkpoint-evaluation.py
```

---

## Evaluation Datasets

Each experiment directory contains a `data/eval/` folder with **pre-computed feature files** for evaluation on:

* **DP**
* **FIVEPILS**
* **MMFakeBench**
* **VERITE**

These files contain **pre-extracted multimodal features** (e.g., image embeddings and metadata), allowing evaluation **without recomputing features**.

---

## Example: Collecting Tweet Text and Image Data (X API)

The CRAVE dataset was constructed using publicly available posts retrieved via the **X (formerly Twitter) API**, including tweet text and associated media.

### Requirements

```bash
pip install tweepy requests
```

### Example Code

```python
import tweepy
import requests
import os
import json

BEARER_TOKEN = "YOUR_BEARER_TOKEN"

client = tweepy.Client(bearer_token=BEARER_TOKEN)

tweet_id = "1234567890123456789"

response = client.get_tweet(
    tweet_id,
    expansions=["attachments.media_keys"],
    media_fields=["url"],
    tweet_fields=["text", "created_at"]
)

tweet = response.data
media = {m.media_key: m for m in response.includes.get("media", [])}

os.makedirs("tweet_data", exist_ok=True)

tweet_record = {
    "tweet_id": tweet_id,
    "text": tweet.text,
    "created_at": str(tweet.created_at),
    "images": []
}

if tweet.attachments:
    for i, key in enumerate(tweet.attachments["media_keys"]):
        image_url = media[key].url

        image_filename = f"{tweet_id}_{i}.jpg"
        image_path = f"tweet_data/{image_filename}"

        img_data = requests.get(image_url).content
        with open(image_path, "wb") as f:
            f.write(img_data)

        tweet_record["images"].append(image_filename)

json_filename = f"tweet_data/{tweet_id}.json"
with open(json_filename, "w") as f:
    json.dump(tweet_record, f, indent=2)

print(f"Saved tweet metadata to {json_filename}")
```

### Output Structure

```
tweet_data/
├── 1234567890123456789.json
├── 1234567890123456789_0.jpg
├── 1234567890123456789_1.jpg
```

The **tweet ID is used as a consistent identifier** for both text and images.

---

## Notes

* The dataset contains **evaluation feature files only**, not raw social media data
* Feature files are stored in `.npy` format for fast reproducibility
* Model checkpoints correspond to the **best validation performance**
* Raw social media content may require rehydration via official APIs where applicable

---

## Citation

If you find this work useful for your research, please cite our preprint. This citation will be updated to reflect the full IEEE Transactions on Computational Social Systems (TCSS) version upon its final publication.

```bibtex
@article{dey2025fact,
  title={Fact-checking with Contextual Narratives: Leveraging Retrieval-Augmented LLMs for Social Media Analysis},
  author={Dey, Arka Ujjal and Awan, Muhammad Junaid and Channing, Georgia and de Witt, Christian Schroeder and Collomosse, John},
  journal={arXiv preprint arXiv:2504.10166},
  year={2025}
}
```

