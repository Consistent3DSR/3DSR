import os
from PIL import Image
import numpy as np
import argparse

def apply_alpha_to_images(folder1, folder2, output_folder):
    """
    Apply alpha channel from images in folder1 to corresponding images in folder2.

    Parameters:
        folder1 (str): Path to the folder containing images with the alpha channel.
        folder2 (str): Path to the folder containing images to receive the alpha channel.
        output_folder (str): Path to save the processed images.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Get list of images in both folders
    images1 = sorted([f for f in os.listdir(folder1) if f.lower().endswith(('png', 'jpg', 'jpeg'))])
    images2 = sorted([f for f in os.listdir(folder2) if f.lower().endswith(('png', 'jpg', 'jpeg'))])

    # Loop through the images in folder1
    for img_name in images1:
        if img_name in images2:  # Ensure the image exists in both folders
            img1_path = os.path.join(folder1, img_name)
            img2_path = os.path.join(folder2, img_name)
            output_path = os.path.join(output_folder, img_name)

            # Open images
            img1 = Image.open(img1_path).convert("RGBA")
            img2 = Image.open(img2_path).convert("RGBA")

            # Resize img1 to match img2
            img1 = img1.resize(img2.size, Image.LANCZOS)

            # Convert images to NumPy arrays
            img1_np = np.array(img1)
            img2_np = np.array(img2)

            # Extract alpha channel from img1 and apply to img2
            alpha_channel = img1_np[:, :, 3]
            img2_np[:, :, 3] = alpha_channel

            # Convert back to PIL image
            img2_alpha_applied = Image.fromarray(img2_np, mode="RGBA")

            # Save output
            # import pdb; pdb.set_trace()
            img2_alpha_applied.save(output_path)
            print(f"✅ Processed: {img_name} -> Saved at {output_path}")

    print("\n🎉 All images processed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply alpha channel from one folder to another.")
    parser.add_argument("--folder1", type=str, required=True, help="Path to folder with alpha channel images.")
    parser.add_argument("--folder2", type=str, required=True, help="Path to folder where alpha will be applied.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to save output images.")
    
    args = parser.parse_args()
    apply_alpha_to_images(args.folder1, args.folder2, args.output_folder)