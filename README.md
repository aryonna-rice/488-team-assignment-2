# Team Assignment 2: Predicting ROI Based on Accepted Loan Data

## Overview
In this project, we were tasked with developing a data-driven investment strategy for Dr. D., who is looking to invest $10,000,000 in peer-to-peer lending on LendingClub.com. Utilizing millions of records of accepted loans from LendingClub.com, spanning from 2007 to 2018, our goal was to propose an investment strategy that categorizes potential returns on investment (ROI) as High, Medium, or Low. This strategy is based on a thorough analysis of various factors including loan default probabilities, value and return of loans, loan duration, and payment delinquencies.

## Project Directory Structure
├── 488env
├── cross-val-results
│   └── 📜 rfc_reduced_results.txt
├── data
│   ├── LendingClub
│   │   ├── raw_data
│   │   │   ├── 📜 features.joblib
│   │   │   └── 📜 target.joblib
│   │   └── train_data
│   │       ├── 📜 features.joblib
│   │       └── 📜 target.joblib
│   └── 📜 selected_features.json
├── imputers
│   ├── 📜 earliest_cr_line.py
│   └── 📜 numeric_data.py
├── models
│   └── 📜 rfc_reduced_data.joblib
├── notebooks
│   └── 📜 roi-classification-notebook.ipynb
├── slurm-scripts
│   ├── 📜 cross_val.sh (SLURM script for cross-validation)
│   └── 📜 rfc.sh (SLURM script for RandomForestClassifier training)
├── training-scripts
│   ├── 📜 cross_val.py (Main script for cross-validation)
│   └── 📜 rfc.py (Main script for RandomForestClassifier training)
└── utils
    └── 📜 utility.py


/pretrains
┣ 📂 models
┃   ┗ 📜 config.yaml
┃   ┗ 📜 v1-5-pruned.ckpt

┣ 📂 generation  
┃   ┗ 📜 checkpoint_best.pth 

┣ 📂 eeg_pretain
┃   ┗ 📜 checkpoint.pth  (pre-trained EEG encoder)

/datasets
┣ 📂 imageNet_images (subset of Imagenet)

┗  📜 block_splits_by_image_all.pth
┗  📜 block_splits_by_image_single.pth 
┗  📜 eeg_5_95_std.pth  

/code
┣ 📂 sc_mbm
┃   ┗ 📜 mae_for_eeg.py
┃   ┗ 📜 trainer.py
┃   ┗ 📜 utils.py

┣ 📂 dc_ldm
┃   ┗ 📜 ldm_for_eeg.py
┃   ┗ 📜 utils.py
┃   ┣ 📂 models
┃   ┃   ┗ (adopted from LDM)
┃   ┣ 📂 modules
┃   ┃   ┗ (adopted from LDM)

┗  📜 stageA1_eeg_pretrain.py   (main script for EEG pre-training)
┗  📜 eeg_ldm.py    (main script for fine-tuning stable diffusion)
┗  📜 gen_eval_eeg.py               (main script for generating images)

┗  📜 dataset.py                (functions for loading datasets)
┗  📜 eval_metrics.py           (functions for evaluation metrics)
┗  📜 config.py                 (configurations for the main scripts)




