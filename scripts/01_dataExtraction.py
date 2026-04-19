# pip install mne numpy

import mne
from mne.datasets import eegbci
import time

# We are targeting the first 20 subjects. 
# Runs 4, 8, 12 correspond strictly to Left/Right fist motor imagery.
subjects = range(1, 21) 
runs = [4, 8, 12]

print(f"Initiating bulk download for {len(subjects)} subjects...")
print("This will download directly to your local ~/mne_data/ directory.")

successful_downloads = 0
failed_subjects = []

for subject_id in subjects:
    try:
        print(f"\n--- Processing Subject {subject_id} ---")
        # load_data automatically downloads and caches the EDF files
        raw_fnames = eegbci.load_data(subject_id, runs)
        
        # Load just the first run to verify the file isn't corrupted
        raw = mne.io.read_raw_edf(raw_fnames[0], preload=False, verbose='ERROR')
        
        successful_downloads += 1
        print(f"Subject {subject_id} secured.")
        
        # Brief pause to avoid hammering the PhysioNet servers
        time.sleep(1) 
        
    except Exception as e:
        print(f"Error downloading Subject {subject_id}: {e}")
        failed_subjects.append(subject_id)

print("\n=== BULK ACQUISITION COMPLETE ===")
print(f"Successfully secured data for {successful_downloads}/{len(subjects)} subjects.")
if failed_subjects:
    print(f"Failed to download subjects: {failed_subjects}")
print("=================================")