scenes=(
    Family
)

for scene in "${scenes[@]}"; do
    echo "Processing scene: $scene"
    # Create the directory if it doesn't exist
    colmap image_undistorter --image_path /fs/nexus-projects/dyn3Dscene/Codes/datasets/TanksAndTempleBG/${scene}/rgb \
                            --input_path /fs/nexus-projects/dyn3Dscene/Codes/datasets/TanksAndTempleBG/${scene}/colmap/sparse/0 \
                            --output_path  /fs/nexus-projects/dyn3Dscene/Codes/datasets/TanksAndTempleBG/processed/${scene} \
                            --output_type COLMAP     
    echo "Done processing scene: $scene"
done
