import os
import json
import argparse
from random import seed
import sys

import h5py
import numpy as np
from PIL import Image
import imageio.v2 as imageio

Image.MAX_IMAGE_PIXELS = None

# ─── Change 1: point to the “missed” file you want to re‐process ───
MISSED_LIST_PATH = '/data/npl/ICEK/ICECAP/icecap-/ViWiki_datamissed.txt'

# ─── (Optional) if you want to write new failures to a separate file ───
RETRY_MISSED_PATH = '/data/npl/ICEK/ICECAP/icecap-/ViWiki_missed_retry.txt'

# ----------------------------------------------------------------------
# Ensure this directory exists so that each resized .jpg can be saved there
RESIZED_DIR = "/data/npl/ICEK/DATASET/images_resized"
os.makedirs(RESIZED_DIR, exist_ok=True)
# ----------------------------------------------------------------------

def resize_image(image_array, size):
    if isinstance(size, int):
        size = (size, size)
    pil_image = Image.fromarray(image_array)
    return np.array(pil_image.resize(size, Image.Resampling.LANCZOS))


def main(params, prefix):
    # ─── Load the list of “just the image names” you want to process ───
    with open(MISSED_LIST_PATH, 'r') as f:
        # each line should be e.g. “0000009544.jpg”
        wanted = { line.strip() for line in f if line.strip() }
    if not wanted:
        print(f"No filenames found in {MISSED_LIST_PATH}; nothing to do.")
        return

    # Hard‐coded JSON path under prefix
    input_json_path = '/data/npl/ICEK/ICECAP/icecap-/' + params['dataset'] + '_data/' + \
                      params['dataset'] + '_cap_basic.json'
    imgs = json.load(open(input_json_path, 'r'))['images']

    seed(123)

    # ─── Open a fresh retry‐missed file ───
    missed_writer = open(RETRY_MISSED_PATH, 'w')
    missed_num = 0

    N = len(imgs)
    print(f"Hello (found {N} total images in JSON; but will only retry {len(wanted):,} of them)")

    # Hard‐coded HDF5 path under prefix
    output_h5_path = '/data/npl/ICEK/ICECAP/icecap-/' + params['dataset'] + '_data/' + \
                     params['dataset'] + '_image.h5'
    f = h5py.File(output_h5_path, "a")  
    # ─── NB: opened in “append” mode so we don't clobber the existing HDF5 ───
    dset = f['images']    # assume the dataset already exists with shape (N,3,256,256)

    for i, img in enumerate(imgs):
        # img['file_path'] is like "somefolder/0000009544.jpg"
        img_name = os.path.basename(img['file_path'])  # e.g. "0000009544.jpg"

        # ─── Change 2: skip everything except those in MISSED_LIST_PATH ───
        if img_name not in wanted:
            continue

        # Hard‐coded source folder
        real_img_path = os.path.join('/data/npl/ICEK/DATASET/images/', img_name)

        try:
            I = imageio.imread(real_img_path)
        except Exception as e:
            print(f"Failed to read image: {real_img_path}. Error: {e}")
            missed_num += 1
            missed_writer.write(img_name + '\n')
            continue

        try:
            Ir = resize_image(I, 256)    # Ir is (256,256) or (256,256,C)
        except Exception as e:
            print(f"Failed resizing image {real_img_path}. Error: {e}")
            missed_num += 1
            missed_writer.write(img_name + '\n')
            continue

        # ─── Drop alpha if it exists ───
        if Ir.ndim == 3 and Ir.shape[2] == 4:
            Ir = Ir[:, :, :3]            # Keep only R,G,B

        # ─── Handle grayscale or other channel counts ───
        if Ir.ndim == 2:
            Ir = np.stack([Ir, Ir, Ir], axis=2)

        elif Ir.ndim == 3:
            C = Ir.shape[2]
            if C == 1:
                Ir = np.concatenate([Ir, Ir, Ir], axis=2)
            elif C == 2:
                gray = Ir[:, :, 0]
                Ir = np.stack([gray, gray, gray], axis=2)
            elif C == 3:
                pass
            else:
                Ir = Ir[:, :, :3]
        else:
            Ir = Ir.reshape(256, 256, -1)[:, :, :3]

        # At this point, Ir.shape == (256, 256, 3)
        Ir_hw3 = Ir.copy()                   # Keep H×W×3 for saving JPEG
        Ir_chw = Ir_hw3.transpose(2, 0, 1)   # Convert to (3,256,256) for HDF5

        # Write into HDF5 at the correct index i
        try:
            dset[i] = Ir_chw
        except Exception as e:
            print(f"Error writing to HDF5 at index {i}: {e}")
            missed_num += 1
            missed_writer.write(img_name + '\n')
            continue

        # ─── Save same (256×256×3) array as a .jpg under RESIZED_DIR ───
        out_path = os.path.join(RESIZED_DIR, img_name)
        try:
            Image.fromarray(Ir_hw3).save(out_path, format='JPEG')
        except Exception as e:
            print(f"Failed to save JPEG {out_path}: {e}")
            # We do not mark this as “missed” because HDF5 succeeded

        # Optional: progress output every 100
        if i % 100 == 0:
            sys.stdout.write(
                f"\rprocessing index {i} (file {img_name}), missed so far {missed_num}"
            )
            sys.stdout.flush()

    f.close()
    missed_writer.close()
    print(f"\nDone retrying. New failures: {missed_num} (listed in {RETRY_MISSED_PATH})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        default='ViWiki',
        choices=['breakingnews', 'goodnews', 'ViWiki']
    )
    prefix = '/data/npl/ICEK/ICECAP/icecap-/ViWiki_data/'
    args = parser.parse_args()
    params = vars(args)
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params, prefix)
