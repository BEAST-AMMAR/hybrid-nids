import urllib.request
import os

url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"
output_file = "/Volumes/Projects/Hybrid 2/data/KDDTest+.txt"

print(f"Downloading NSL-KDD Test+ dataset from {url}...")
try:
    urllib.request.urlretrieve(url, output_file)
    print(f"Dataset successfully downloaded to {output_file}")
except Exception as e:
    print(f"Error downloading dataset: {e}")
