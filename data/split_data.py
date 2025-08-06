import os
import shutil
import random

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# Define paths
image_folder_path = os.path.join(SCRIPT_DIR, "classes")
debug_folder = os.path.join(SCRIPT_DIR, "debug")
train_folder = os.path.join(SCRIPT_DIR, "train")
val_folder = os.path.join(SCRIPT_DIR, "val")
test_folder = os.path.join(SCRIPT_DIR, "test")
valtest_folder = os.path.join(SCRIPT_DIR, "val_test")
fulldata_folder = os.path.join(SCRIPT_DIR, "full_data")

# Create target folders
for folder in [
    debug_folder,
    train_folder,
    val_folder,
    test_folder,
    valtest_folder,
    fulldata_folder,
]:
    os.makedirs(folder, exist_ok=True)

# Copy full data
shutil.copytree(image_folder_path, os.path.join(fulldata_folder, "classes"), dirs_exist_ok=True)

# Print folder paths
print(debug_folder)
print(train_folder)
print(val_folder)
print(test_folder)
print(valtest_folder)

# Clear target folders
for folder in [
    debug_folder,
    train_folder,
    val_folder,
    test_folder,
    valtest_folder,
]:
    for root, dirs, files in os.walk(folder):
        for file in files:
            os.remove(os.path.join(root, file))

# Process each class
for class_name in os.listdir(image_folder_path):
    class_folder = os.path.join(image_folder_path, class_name)
    if not os.path.isdir(class_folder):
        continue

    print(class_name)

    # Create subfolders
    for base in [
        debug_folder,
        train_folder,
        val_folder,
        test_folder,
        valtest_folder,
    ]:
        os.makedirs(os.path.join(base, class_name), exist_ok=True)

    # Get file list
    files = [f for f in os.listdir(class_folder) if os.path.isfile(os.path.join(class_folder, f))]
    total_files = len(files)

    # Set ratios
    debug_ratio = 0.05
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1

    # Compute split sizes
    debug_count = round(debug_ratio * total_files)
    train_count = round(train_ratio * total_files)
    val_count = round(val_ratio * total_files)

    # Shuffle files deterministically
    random.Random(42).shuffle(files)

    # Split files
    debug_files = files[:debug_count]
    train_files = files[debug_count:debug_count + train_count]
    val_files = files[debug_count + train_count:debug_count + train_count + val_count]
    test_files = files[debug_count + train_count + val_count:]

    # Copy debug files
    for f in debug_files:
        shutil.copy(os.path.join(class_folder, f), os.path.join(debug_folder, class_name, f))

    # Move train files
    for f in train_files:
        shutil.move(os.path.join(class_folder, f), os.path.join(train_folder, class_name, f))

    # Copy remaining to valtest
    for f in os.listdir(class_folder):
        shutil.copy(os.path.join(class_folder, f), os.path.join(valtest_folder, class_name, f))

    # Move val files
    for f in val_files:
        shutil.move(os.path.join(class_folder, f), os.path.join(val_folder, class_name, f))

    # Move remaining to test
    for f in os.listdir(class_folder):
        shutil.move(os.path.join(class_folder, f), os.path.join(test_folder, class_name, f))

# Cleanup
shutil.rmtree(image_folder_path)
print("Images divided into train, validation, and test splits successfully!")
