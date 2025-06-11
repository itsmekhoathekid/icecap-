import requests

url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.vi.300.bin.gz"
output_path = "cc.vi.300.bin.gz"

print("Downloading cc.vi.300.bin.gz...")
with requests.get(url, stream=True) as r:
    r.raise_for_status()
    with open(output_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
print(f"Download complete: {output_path}")

print("Extracting...")
import gzip
import shutil

with gzip.open(output_path, 'rb') as f_in:
    with open('cc.vi.300.bin', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
print("Extracted to cc.vi.300.bin")
