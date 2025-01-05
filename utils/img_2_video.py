import cv2
import os
import tqdm

def images_to_video(image_folder, video_filename, fps=30):
    # Get all image files from the folder
    images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
    images.sort()  # Sort to maintain order

    # Read the first image to get the video frame dimensions
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    video = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        
        if frame is None:
            print(f"Warning: Couldn't read image {image_path}")
            continue

        video.write(frame)

    # Release the VideoWriter
    video.release()
    print(f"Video saved as {video_filename}")

# Example usage:
image_folder = "/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting/frames/interpolated_30"
video_filename = os.path.join(image_folder, "video.mp4")
images_to_video(image_folder, video_filename)
