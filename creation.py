import os
import numpy as np
import pandas as pd
import mne
from scipy.stats import kurtosis
from numpy.linalg import norm

# Load demographic data from participants.tsv
participants_df = pd.read_csv('ds004504/participants.tsv', sep='\t')
# Ensure column names are lower case for consistency
participants_df.columns = [col.lower() for col in participants_df.columns]

# Subject lists
ad_subjects = [f"sub-{i:03d}" for i in range(1, 38)]
cn_subjects = [f"sub-{i:03d}" for i in range(38, 66)]

base_path = 'ds004504/derivatives'

all_data = []

for subject in ad_subjects + cn_subjects:
    label = 1 if subject in ad_subjects else 0
    eeg_file = os.path.join(base_path, subject, 'eeg', f'{subject}_task-eyesclosed_eeg.set')

    if not os.path.exists(eeg_file):
        print(f"❌ File not found: {eeg_file}")
        continue

    # Get demographic info for this subject
    demo_row = participants_df[participants_df['participant_id'] == subject]
    if demo_row.empty:
        print(f"⚠ Demographic info not found for {subject}")
        continue
    age = demo_row['age'].values[0]
    sex = demo_row['gender'].values[0]  # Assuming 'gender' is the column name for sex

    try:
        print(f"✅ Processing {subject}...")
        raw = mne.io.read_raw_eeglab(eeg_file, preload=True)
        raw.filter(0.5, 45., fir_design='firwin')
        raw.set_montage('standard_1020')
        
        # Set epoch duration to 10 seconds
        epochs = mne.make_fixed_length_epochs(raw, duration=10, preload=True)

        for epoch in epochs:
            features = []
            for ch_data in epoch:
                lbp = np.log(np.mean(ch_data ** 2) + 1e-6)  # Log Band Power
                var = np.var(ch_data)
                std = np.std(ch_data)
                kur = kurtosis(ch_data)
                ae = np.sum(ch_data ** 2)
                rms = np.sqrt(np.mean(ch_data ** 2))
                l2_norm = norm(ch_data)
                features.extend([lbp, var, std, kur, ae, rms, l2_norm])
            # Add label, age, and sex for each epoch
            features.extend([label, age, sex])
            all_data.append(features)

    except Exception as e:
        print(f"⚠ Error processing {subject}: {e}")

# Get channel names and create column names
sample_file = os.path.join(base_path, ad_subjects[0], 'eeg', f'{ad_subjects[0]}_task-eyesclosed_eeg.set')
raw_sample = mne.io.read_raw_eeglab(sample_file, preload=False)
channels = raw_sample.ch_names
feature_names = ['LBP', 'VAR', 'STD', 'KUR', 'AE', 'RMS', 'L2']
columns = [f"{ch}_{f}" for ch in channels for f in feature_names] + ['label', 'age', 'sex']

# Create and save DataFrame
df_all = pd.DataFrame(all_data, columns=columns)
print(df_all.head())

df_all.to_csv("eeg_features_ad_vs_cn_with_demo.csv", index=False)
print("✅ Features with demographics saved to eeg_features_ad_vs_cn_with_demo.csv")