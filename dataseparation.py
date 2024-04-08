import os
import shutil
from sklearn.model_selection import train_test_split

def split_data(source_folder, dest_folder, split_size=0.2):
    # Create training and validation directories
    train_dir = os.path.join(dest_folder, 'train')
    val_dir = os.path.join(dest_folder, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Loop over both classes
    for class_folder in ['noLeaves', 'withLeaves']:
        # Create subdirectories for training and validation
        os.makedirs(os.path.join(train_dir, class_folder), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_folder), exist_ok=True)

        # Full path of the class source folder
        class_dir = os.path.join(source_folder, class_folder + '00041')
        
        # Get a list of filenames
        images = os.listdir(class_dir)
        # Split the images into training and validation sets
        train_images, val_images = train_test_split(images, test_size=split_size, random_state=42)
        
        # Copy images to the train directory
        for img in train_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(train_dir, class_folder, img))
        
        # Copy images to the val directory
        for img in val_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(val_dir, class_folder, img))

# Assuming the source folders are named 'noLeaves00041' and 'withLeaves00041'
split_data('finalData', 'finalData', split_size=0.2)
