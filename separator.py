import os
import shutil

base_path = r"C:\Users\Mikhael\Downloads\data-slayer-3\train"

folders = ["drowsiness", "non-drowsiness"]

for f in folders:
    src_folder = os.path.join(base_path, f)
    dst_folder = os.path.join(base_path, f + "_separated")

    # Make destination folder if not exist
    os.makedirs(dst_folder, exist_ok=True)

    # Loop through subject folders
    for subject in os.listdir(src_folder):
        subject_path = os.path.join(src_folder, subject)

        # Skip non-folder files
        if not os.path.isdir(subject_path):
            continue

        # Loop through videos inside subject folder
        for file in os.listdir(subject_path):
            file_path = os.path.join(subject_path, file)

            # Skip anything that isn't a video
            if not os.path.isfile(file_path):
                continue

            # Create new filename
            name, ext = os.path.splitext(file)
            new_name = f"{name}_{subject}{ext}"
            new_path = os.path.join(dst_folder, new_name)

            # Copy video
            shutil.copy2(file_path, new_path)

print("Done separating videos!")
