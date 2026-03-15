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
Below is a simplified example demonstrating how tweet text and image URLs can be retrieved and stored.

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

# Replace with your credentials from https://developer.twitter.com/
BEARER_TOKEN = "YOUR_BEARER_TOKEN"

client = tweepy.Client(bearer_token=BEARER_TOKEN)

# Tweet ID example
tweet_id = "1234567890123456789"

response = client.get_tweet(
    tweet_id,
    expansions=["attachments.media_keys"],
    media_fields=["url"],
    tweet_fields=["text","created_at"]
)

tweet = response.data
media = {m.media_key: m for m in response.includes["media"]}

print("Tweet text:", tweet.text)

os.makedirs("tweet_data/images", exist_ok=True)

if tweet.attachments:
    for key in tweet.attachments["media_keys"]:
        image_url = media[key].url
        filename = image_url.split("/")[-1]

        img_data = requests.get(image_url).content
        with open(f"tweet_data/images/{filename}", "wb") as f:
            f.write(img_data)

        print("Saved image:", filename)
```

### Output

The script saves:

```
tweet_data/
│
├── images/
│   └── tweet_image.jpg
```

and prints the tweet text to the console.

### Notes

* Only publicly available tweets were accessed.
* Images were downloaded from the media URLs provided by the X API.
* Metadata such as tweet text, timestamps, and media URLs were stored for downstream multimodal misinformation analysis.

Refer to the official X API documentation for authentication and usage details.


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
