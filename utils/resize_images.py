from PIL import Image
from tqdm import tqdm
import os
import argparse
import natsort
import multiprocessing as mp


def resize_img(args):
    input_image_path, output_image_path, resize_ratio, new_w, new_h = args
    with Image.open(input_image_path) as img:
        width, height = img.size

        # Ensure dimensions are divisible by 4
        width -= width % 4
        height -= height % 4

        if new_w != 0 or new_h != 0:
            new_size = (new_w, new_h)
        elif resize_ratio == 1:
            new_size = (width, height)
        else:
            new_size = (width // resize_ratio, height // resize_ratio)

        resized_img = img.resize(new_size, Image.BICUBIC)
        resized_img.save(output_image_path)


def resize_dataset(base_input_dir, base_output_dir, ratios):
    
    input_dir = os.path.join(base_input_dir, "images")
    
    if not os.path.exists(input_dir):
        print(f"⚠️ no 'images' folder found")        

    img_paths = natsort.natsorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('jpg', 'jpeg', 'JPG', 'png'))
    ])

    for ratio in ratios:
        output_dir = os.path.join(base_output_dir, f"images_{ratio}")
        os.makedirs(output_dir, exist_ok=True)

        print(f"Processing with ratio {ratio} using multiprocessing...")
        tasks = []
        for img_name in img_paths:
            img_path = os.path.join(input_dir, img_name)
            out_name = os.path.splitext(img_name)[0] + ".png"
            output_path = os.path.join(output_dir, out_name)
            tasks.append((img_path, output_path, ratio, 0, 0))

        with mp.Pool(mp.cpu_count()) as pool:
            list(tqdm(pool.imap(resize_img, tasks), total=len(tasks)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to the input folder containing all scene folders")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to the output folder containing all scene folders")
    args = parser.parse_args()

    ratios = [2, 4, 8, 16]
    resize_dataset(args.input_dir, args.output_dir, ratios)