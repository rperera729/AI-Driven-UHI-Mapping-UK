import os
import numpy as np
import rioxarray

# --- CONFIGURATION ---
INPUT_DIR = r"D:\UHI_Project\data_processed"
PATCH_DIR = r"D:\UHI_Project\data_processed\patches"
PATCH_SIZE = 128

def generate_all_patches():
    # Create the folder to hold the patches
    os.makedirs(PATCH_DIR, exist_ok=True)
    
    # Find all the 5-layer stacks we just made
    stack_files = [f for f in os.listdir(INPUT_DIR) if f.startswith("UHI_Stack") and f.endswith(".tif")]
    
    total_patches = 0
    
    for file in stack_files:
        print(f"Slicing {file} into patches...")
        path = os.path.join(INPUT_DIR, file)
        
        # Open the stack (5 layers)
        with rioxarray.open_rasterio(path) as src:
            data = src.values # This loads [bands, height, width]
            _, height, width = data.shape
            
            # Use a 'sliding window' to cut the patches
            for y in range(0, height - PATCH_SIZE, PATCH_SIZE):
                for x in range(0, width - PATCH_SIZE, PATCH_SIZE):
                    patch = data[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                    
                    # VALIDATION: Only save if the patch isn't empty (NaN)
                    # and if it doesn't contain errors
                    if not np.isnan(patch).all():
                        patch_filename = f"patch_{total_patches:04d}.npy"
                        np.save(os.path.join(PATCH_DIR, patch_filename), patch)
                        total_patches += 1

    print(f"\n DONE! Created {total_patches} patches in {PATCH_DIR}")

if __name__ == "__main__":
    generate_all_patches()
