import os
import json
import argparse

# ─── Directory where resized images are saved ───
RESIZED_DIR = "/data/npl/ICEK/DATASET/images_resized"

def main(params, prefix):
    # Path to the JSON file containing all image entries
    input_json_path = (
        '/data/npl/ICEK/ICECAP/icecap-/' 
        + params['dataset'] 
        + '_data/' 
        + params['dataset'] 
        + '_cap_basic.json'
    )
    with open(input_json_path, 'r') as f:
        imgs = json.load(f)['images']

    missing = []

    for img in imgs:
        # Extract the filename (e.g., "0000009544.jpg") from "somefolder/0000009544.jpg"
        img_name = os.path.basename(img['file_path'])
        resized_path = os.path.join(RESIZED_DIR, img_name)

        # If the resized version does not exist, record it
        if not os.path.exists(resized_path):
            missing.append(img_name)

    # Print summary
    print(f"Total images listed in JSON: {len(imgs)}")
    print(f"Resized directory: {RESIZED_DIR}")
    print(f"Images NOT found in resized directory: {len(missing)}\n")

    # List them to stdout
    for name in missing:
        print(name)

    # Optionally, write them to a text file under the same prefix
    output_list_path = os.path.join(prefix, 'missing_resized.txt')
    with open(output_list_path, 'w') as out_f:
        for name in missing:
            out_f.write(f"{name}\n")
    print(f"\nWrote missing‐file list to: {output_list_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        default='ViWiki',
        choices=['breakingnews', 'goodnews', 'ViWiki'],
        help="Which dataset JSON to check (uses <dataset>_cap_basic.json)"
    )
    args = parser.parse_args()
    params = vars(args)

    # Example prefix: '/data/npl/ICEK/ICECAP/icecap-/ViWiki_data/'
    prefix = '/data/npl/ICEK/ICECAP/icecap-/' + params['dataset'] + '_data/'

    print("Parsed parameters:")
    print(json.dumps(params, indent=2))
    main(params, prefix)
