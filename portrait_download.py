import os
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm

DATA_DIR = "./opendata/data"
IMAGE_DIR = "./images"
os.makedirs(IMAGE_DIR, exist_ok=True)

df_objects = pd.read_csv(os.path.join(DATA_DIR, "objects.csv"))
df_images = pd.read_csv(os.path.join(DATA_DIR, "published_images.csv"))

allowed_classes = ["Painting", "Drawing", "Print", "Miniature"]

df_portraits = df_objects[
    df_objects["title"].str.contains("portrait", case=False, na=False) &
    df_objects["classification"].isin(allowed_classes) &
    (df_objects["isvirtual"] == 0)
]

df_images_primary = df_images[df_images["viewtype"] == "primary"]
df_merged = df_portraits.merge(
    df_images_primary,
    left_on="objectid",
    right_on="depictstmsobjectid"
)
df_merged = df_merged[df_merged["iiifurl"].notna()]

def download_image(iiif_url, object_id, classification):
    filename = f"{classification.lower()}_{object_id}.jpg"
    save_path = os.path.join(IMAGE_DIR, filename)
    
    if os.path.exists(save_path):
        return True

    final_url = iiif_url + "/full/800,/0/default.jpg"
    try:
        response = requests.get(final_url, timeout=10)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content)).convert("RGB")
            image.save(save_path)
            return True
    except Exception as e:
        print(f"❌ Error downloading {filename}: {e}")
    return False

print(f"Total filtered portraits to download: {len(df_merged)}")

for _, row in tqdm(df_merged.iterrows(), total=len(df_merged), desc="Downloading portraits"):
    download_image(row["iiifurl"], row["objectid"], row["classification"])
