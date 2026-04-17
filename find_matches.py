import os
import glob
import re

# UPDATE THESE PATHS to your exact D: drive locations
LANDSAT_ROOT = r"D:\UHI_Project\data_raw\landsat" # Folders 2023, 2024, 2025 are inside here
SENTINEL_ROOT = r"D:\UHI_Project\data_raw\sentinel" # Unzipped .SEN3 folders here

def find_matches():
    # 1. Recursive search for Landsat Band 10 in all subfolders
    # The '**' means it will look inside 2023, 2024, and 2025 folders automatically
    ls_pattern = os.path.join(LANDSAT_ROOT, "**", "*_ST_B10.TIF")
    ls_files = glob.glob(ls_pattern, recursive=True)

    print(f"Found {len(ls_files)} Landsat scenes across all years.\n")

    for ls_path in ls_files:
        ls_filename = os.path.basename(ls_path)
        
        # Extract the acquisition date (e.g., 20230805)
        # It finds the first 8-digit number in the filename
        match = re.search(r"(\d{8})", ls_filename)
        if match:
            date_str = match.group(1)
            
            # 2. Search for the Sentinel-3 folder matching this date
            s3_pattern = os.path.join(SENTINEL_ROOT, f"*{date_str}*.SEN3")
            s3_matches = glob.glob(s3_pattern)
            
            if s3_matches:
                print(f" MATCH FOUND: {date_str}")
                print(f"   Landsat:  {ls_filename}")
                print(f"   Sentinel: {os.path.basename(s3_matches[0])}")
                print("-" * 30)
            else:
                print(f" No Sentinel match for Landsat date: {date_str}")

if __name__ == "__main__":
    find_matches()
