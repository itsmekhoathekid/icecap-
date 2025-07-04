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

# -----------------------------------------------------------------------------
# Ensure this directory exists so that each resized .jpg can be saved there
RESIZED_DIR = "/data/npl/ICEK/Wikipedia/images_resized"
os.makedirs(RESIZED_DIR, exist_ok=True)
# -----------------------------------------------------------------------------

def resize_image(image_array, size):
    if isinstance(size, int):
        size = (size, size)
    pil_image = Image.fromarray(image_array)
    return np.array(pil_image.resize(size, Image.Resampling.LANCZOS))


def main(params, prefix):
    # Hard-coded JSON path under prefix
    input_json_path = '/data/npl/ICEK/ICECAP/icecap-/' + params['dataset'] + '_data/' + \
                      params['dataset'] + '_cap_basic.json'
    imgs = json.load(open(input_json_path, 'r'))['images']

    seed(123)
    missed_writer = open(prefix + 'missed.txt', 'w')
    missed_num = 0

    N = len(imgs)
    print("Hello (found {} images)".format(N))

    # Hard-coded HDF5 path under prefix
    output_h5_path = '/data/npl/ICEK/ICECAP/icecap-/' + params['dataset'] + '_data/' + \
                     params['dataset'] + '_image.h5'
    f = h5py.File(output_h5_path, "w")
    dset = f.create_dataset("images", (N, 3, 256, 256), dtype='uint8')
    print("Opened HDF5 for writing at:", output_h5_path)

    for i, img in enumerate(imgs):
        # img['file_path'] is expected to be something like "somefolder/0000009544.jpg"
        img_path = img['file_path'].split("/")[1]
        # Hard-coded source folder
        real_img_path = os.path.join('/data/npl/ICEK/Wikipedia/images/', img_path)

        try:
            I = imageio.imread(real_img_path)
        except Exception as e:
            print(f"Failed to read image: {real_img_path}. Error: {e}")
            missed_num += 1
            missed_writer.write(img['file_path'] + '\n')
            continue

        try:
            Ir = resize_image(I, 256)    # Ir is (256,256) or (256,256,C)
        except Exception as e:
            print(f"Failed resizing image {real_img_path}. Error: {e}")
            missed_num += 1
            missed_writer.write(img['file_path'] + '\n')
            continue

        # ── Drop alpha if it exists ──
        if Ir.ndim == 3 and Ir.shape[2] == 4:
            Ir = Ir[:, :, :3]            # Keep only R,G,B

        # ── Handle grayscale or other channel counts ──
        if Ir.ndim == 2:
            # pure grayscale → (256,256) → replicate to (256,256,3)
            Ir = np.stack([Ir, Ir, Ir], axis=2)

        elif Ir.ndim == 3:
            C = Ir.shape[2]
            if C == 1:
                # single-channel → replicate
                Ir = np.concatenate([Ir, Ir, Ir], axis=2)
            elif C == 2:
                # two channels → treat channel 0 as gray, replicate it
                gray = Ir[:, :, 0]
                Ir = np.stack([gray, gray, gray], axis=2)
            elif C == 3:
                # already RGB → do nothing
                pass
            else:
                # C ≥ 4 → drop extras
                Ir = Ir[:, :, :3]

        else:
            # Unexpected ndim > 3 → flatten to first 3
            Ir = Ir.reshape(256, 256, -1)[:, :, :3]

        # At this point, Ir.shape == (256, 256, 3)
        Ir_hw3 = Ir.copy()                   # Keep H×W×3 for saving JPEG
        Ir_chw = Ir_hw3.transpose(2, 0, 1)   # Convert to (3,256,256) for HDF5

        # Write the (3,256,256) array into HDF5
        try:
            dset[i] = Ir_chw
        except Exception as e:
            print(f"Error writing to HDF5 at index {i}: {e}")
            missed_num += 1
            missed_writer.write(img['file_path'] + '\n')
            continue

        # ── Save the same (256×256×3) array as a .jpg under RESIZED_DIR ──
        out_path = os.path.join(RESIZED_DIR, img_path)
        try:
            Image.fromarray(Ir_hw3).save(out_path, format='JPEG')
        except Exception as e:
            print(f"Failed to save JPEG {out_path}: {e}")
            # Not marking as “missed” because HDF5 succeeded

        if i % 1000 == 0:
            sys.stdout.write(
                f"\rprocessing {i}/{N} ({i * 100.0 / N:.2f}% done) missed {missed_num}"
            )
            sys.stdout.flush()

    f.close()
    missed_writer.close()
    print("\nDone. Wrote HDF5 to", output_h5_path)
    print("Missed {} images (listed in {}missed.txt)".format(missed_num, prefix))


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
