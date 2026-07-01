# ======================================================
# PATH TO THE OPENNEURO EEG DATASET
#
# Download the dataset from OpenNeuro and update the
# path below to match your local installation.
# Then, change this path to the location of the OpenNeuro Data:
#
# main_folder = r"C:\Users\angel\Documents\GHIBLI\ENIAC 2026"
# ======================================================



import os
import mne
import numpy as np

# --------------------------------------------
# MAIN FOLDER
# --------------------------------------------

main_folder = r"C:\Users\angel\Documents\GHIBLI\ENIAC 2026"

# --------------------------------------------
# OUTPUT FOLDER
# --------------------------------------------

output_folder = os.path.join(
    main_folder,
    "connectivity_matrices"
)

os.makedirs(output_folder, exist_ok=True)

# --------------------------------------------
# PROCESS ALL SUBJECTS
# --------------------------------------------

for subject_number in range(1, 61):

    subject_name = f"sub-{subject_number:02d}"

    print("\n========================")
    print(subject_name)
    print("========================")

    subject_folder = os.path.join(
        main_folder,
        subject_name
    )

    conditions = {

        "open":
        f"{subject_name}_ses-session1_task-eyesopen_eeg.vhdr",

        "closed":
        f"{subject_name}_ses-session1_task-eyesclosed_eeg.vhdr"

    }

    for label, filename in conditions.items():

        vhdr_file = os.path.join(
            subject_folder,
            filename
        )

        print("Loading:", filename)

        raw = mne.io.read_raw_brainvision(
            vhdr_file,
            preload=True,
            verbose=False
        )

        # ------------------------------------
        # FILTER
        # ------------------------------------

        raw.filter(
            l_freq=1,
            h_freq=40,
            verbose=False
        )

        # ------------------------------------
        # DATA
        # ------------------------------------

        data = raw.get_data()

        # ------------------------------------
        # CONNECTIVITY MATRIX
        # ------------------------------------

        connectivity_matrix = np.corrcoef(data)

        # ------------------------------------
        # SAVE MATRIX
        # ------------------------------------

        save_name = f"{subject_name}_{label}.npy"

        save_path = os.path.join(
            output_folder,
            save_name
        )

        np.save(
            save_path,
            connectivity_matrix
        )

        print("Saved:", save_name)

print("\nDONE!")

