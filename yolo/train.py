import os
from tqdm import tqdm
from process_data import extract_polygones
import shutil

def process_data(masks_path, output_dir, classe_dict):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "train", "labels"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val", "labels"), exist_ok=True)

    # Loop through train, val, and test splits
    splits = ['train', 'val']
    for split in splits:
        split_dir = os.path.join(masks_path, split)
        print(f"Processing {split} split...")

        # Count the total number of files for the progress bar
        total_files = sum(
            1 for city in os.listdir(split_dir) 
            for file in os.listdir(os.path.join(split_dir, city)) 
            if file.endswith('_gtFine_instanceIds.png')  # Use instance IDs file
        )
        
        with tqdm(total=total_files, desc=f"{split} split") as pbar:
            for city in os.listdir(split_dir):
                city_dir = os.path.join(split_dir, city)
                for file in os.listdir(city_dir):
                    if file.endswith('_gtFine_instanceIds.png'):

                        # Input mask path
                        mask_path = os.path.join(city_dir, file)

                        yolo_annotations = extract_polygones(mask_path, classe_dict=classe_dict)
                        # Save YOLO annotation file
                        if yolo_annotations:
                            yolo_file = file.replace('_gtFine_instanceIds.png', '.txt')
                            yolo_path = os.path.join(output_dir, split, "labels")
                            os.makedirs(yolo_path, exist_ok=True)

                            with open(os.path.join(yolo_path, yolo_file), 'w') as f:
                                for yolo_class_id, polygon in yolo_annotations:
                                    f.write(f"{yolo_class_id} ")
                                    f.write(" ".join(map(str, polygon)))
                                    f.write("\n")
                        pbar.update(1)


def move_images(images_path, output_dir):

    # Ensure destination directories exist
    os.makedirs(os.path.join(output_dir, "train", "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val", "images"), exist_ok=True)

    # Loop through train and val splits
    splits = ['train', 'val']
    for split in splits:
        source_split_dir = os.path.join(images_path, split)
        destination_split_dir = os.path.join(output_dir, split, "images")

        # Collect all image paths for progress tracking
        all_images = [
            os.path.join(source_split_dir, city, file)
            for city in os.listdir(source_split_dir)
            if os.path.isdir(os.path.join(source_split_dir, city))  # Ensure it's a city folder
            for file in os.listdir(os.path.join(source_split_dir, city))
            if file.endswith('.png') or file.endswith('.jpg')
        ]

        # Copy images directly to the destination
        with tqdm(total=len(all_images), desc=f"Copying {split} images") as pbar:
            for image_path in all_images:
                file_name = os.path.basename(image_path)
                # Replace '_leftImg8bit.png' with '.png'
                new_file_name = file_name.replace('_leftImg8bit.png', '.png')
                destination_path = os.path.join(destination_split_dir, new_file_name)

                shutil.copy(image_path, destination_path)
                pbar.update(1)

    print("Images copied successfully.")



if __name__ == '__main__':
    # masks paths 
    masks_path = "/kaggle/input/cityscapes/Cityspaces/gtFine"
    output_dir = "/kaggle/working/data"


    # imgs paths
    images_path = "/kaggle/input/cityscapes/Cityspaces/images"

    # classes didicated to instance segmentation
    native_ids = {
        'person': 24,
        'rider': 25,
        'car': 26,
        'truck': 27,
        'bus': 28,
        'caravan': 29,
        'trailer': 30,
        'train': 31,
        'motorcycle': 32,
        'bicycle': 33,
    }
    yolo_train_ids = {
        'person': 0,
        'rider': 1,
        'car': 2,
        'truck': 3,
        'bus': 4,
        'caravan': 5,
        'trailer': 6,
        'train': 7,
        'motorcycle': 8,
        'bicycle': 9,
    }

    # Reverse mapping for easy lookup
    classe_dict = {v: yolo_train_ids[k] for k, v in native_ids.items()}
    process_data(masks_path, output_dir, classe_dict)
    move_images(images_path, output_dir)