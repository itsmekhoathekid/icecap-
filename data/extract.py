import zipfile
import os

zip_path = "/data/npl/ICEK/ICECAP/icecap-/data/word2vec_vi_syllables_300dims.zip"
extract_dir = os.path.dirname(zip_path)  # giải nén ra cùng thư mục với file zip

print(f"Extracting {zip_path} to {extract_dir} ...")

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
    print("Done. Extracted files:")
    for file_name in zip_ref.namelist():
        print(" -", file_name)

print("All files extracted successfully!")
