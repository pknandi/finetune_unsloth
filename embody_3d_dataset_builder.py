import os
import random
import shutil


def pick_and_copy_matching_dirs(parent_dir, secondary_dir, dest_dir1, dest_dir2, num_picks):
    """
    Randomly picks child directories from parent_dir.
    Skips if already in destination.
    If a matching directory exists in secondary_dir, it copies both to their respective destinations.
    """
    # 1. Get a list of all items in parent_dir that are actually directories
    try:
        all_children = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
    except FileNotFoundError:
        print(f"❌ Error: The parent directory '{parent_dir}' does not exist.")
        return

    # 2. Handle edge cases
    if not all_children:
        print("⚠️ No child directories found in the parent directory.")
        return

    # 3. Ensure the destination directories exist (creates them if they don't)
    os.makedirs(dest_dir1, exist_ok=True)
    os.makedirs(dest_dir2, exist_ok=True)

    # 4. Shuffle the list so we can iterate randomly until we hit our quota
    random.shuffle(all_children)

    copied_count = 0

    # 5. Loop through the randomized list
    for child_name in all_children:
        # Stop if we've reached the requested number of picks
        if copied_count >= num_picks:
            break

        parent_child_path = os.path.join(parent_dir, child_name)
        secondary_child_path = os.path.join(secondary_dir, child_name)
        dest1_path = os.path.join(dest_dir1, child_name)
        dest2_path = os.path.join(dest_dir2, child_name)

        # --- NEW LOGIC: Check if it already exists in the destination ---
        if os.path.exists(dest1_path) or os.path.exists(dest2_path):
            print(f"⏭️  Skipping '{child_name}': Already exists in destination.")
            continue

        # Check if a directory with the exact same name exists in the secondary directory
        if os.path.exists(secondary_child_path) and os.path.isdir(secondary_child_path):
            print(f"✅ Match found for: '{child_name}'")

            # Copy from Parent to Destination 1
            try:
                shutil.copytree(parent_child_path, dest1_path, dirs_exist_ok=True)
                print(f"   -> Copied from Parent to: {dest1_path}")
            except Exception as e:
                print(f"   -> ❌ Error copying to {dest1_path}: {e}")

            # Copy from Secondary to Destination 2
            try:
                shutil.copytree(secondary_child_path, dest2_path, dirs_exist_ok=True)
                print(f"   -> Copied from Secondary to: {dest2_path}")
            except Exception as e:
                print(f"   -> ❌ Error copying to {dest2_path}: {e}")

            # Successfully copied, increment the counter
            copied_count += 1
        else:
            # Uncomment the line below if you want to see logs for folders that didn't have a secondary match
            # print(f"❌ No match found in secondary directory for: '{child_name}'")
            pass

    # 6. Final Summary
    print(f"\n🏁 Finished! Successfully copied {copied_count} new directories.")
    if copied_count < num_picks:
        print(f"⚠️ Note: Only {copied_count} out of {num_picks} requested directories were copied (ran out of valid matches or remaining folders).")


# ==========================================
# Configuration & Execution
# ==========================================
if __name__ == "__main__":
    PARENT_DIR = "./embody-3d/embody_3d__20251016__SCENARIOS_audio/acting"
    SECONDARY_DIR = "./embody-3d/embody_3d__20251016__SCENARIOS_smplx/acting"

    PARENT_DIR = "./embody-3d/embody_3d__20251016__DAYLIFE_audio/daylife"
    SECONDARY_DIR = "./embody-3d/embody_3d__20251016__DAYLIFE_smplx/daylife"

    DESTINATION_1 = "./embody-3d/training_dataset/audio"
    DESTINATION_2 = "./embody-3d/training_dataset/smplx"

    NUMBER_OF_PICKS = 20

    pick_and_copy_matching_dirs(
        parent_dir=PARENT_DIR,
        secondary_dir=SECONDARY_DIR,
        dest_dir1=DESTINATION_1,
        dest_dir2=DESTINATION_2,
        num_picks=NUMBER_OF_PICKS,
    )
