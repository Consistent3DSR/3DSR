#!/bin/bash

FOLDER_A="/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/independent_SR/StableSR/original_setting/input_DS_2/bicycle/ours_30000/DS_2/test_preds_2"
FOLDER_B="/fs/nexus-projects/dyn3Dscene/Codes/data/my_new_resize/bicycle/images_2"
FOLDER_C="/fs/nexus-projects/dyn3Dscene/Codes/mip-splatting-mine/outputs/independent_SR/StableSR/original_setting/input_DS_2/bicycle/ours_30000/DS_2/gt_2"

mkdir -p "$FOLDER_C"  # Create folder_C if it doesn't exist

for file in "$FOLDER_A"/*; do
    filename=$(basename "$file")
    if [ -f "$FOLDER_B/$filename" ]; then
        cp "$file" "$FOLDER_C/$filename"
    fi
done