# Team Assignment 2: Predicting ROI Based on Accepted Loan Data

## Overview
In this project, we were tasked with developing a data-driven investment strategy for Dr. D., who is looking to invest $10,000,000 in peer-to-peer lending on LendingClub.com. Utilizing millions of records of accepted loans from LendingClub.com, spanning from 2007 to 2018, our goal was to propose an investment strategy that categorizes potential returns on investment (ROI) as High, Medium, or Low. This strategy is based on a thorough analysis of various factors including loan default probabilities, value and return of loans, loan duration, and payment delinquencies.

## Project Directory Structure
The contents of the data folder and models folder are not included in this repository. Please download them from the team's drive `[here] (https://drive.google.com/drive/folders/19UleGzYyksu3571awwAyb2PPmgxvzLij?usp=drive_link)` and put them in the correct places in this repository as shown below. 
* Add image here *

## Environment Setup
This project was developed using Python and Jupyter notebooks. To replicate our environment, please follow these steps:

1. Ensure you have Python installed on your system.
2. Create a virtual environment named 488env by running python -m venv 488env in your terminal.
3. Activate the virtual environment:
    - On Windows, use 488env\Scripts\activate.
    - On Unix or MacOS, use source 488env/bin/activate.
4. Install the required dependencies by running pip install -r requirements.txt.

Note: All model training and heavy computations were performed using UNC's Longleaf services due to the extensive computational resources required. It is recommended to use these services or similar for training the models, as local machines may not be suitable.

## Usage
To analyze the data and review the investment strategy, open the Jupyter notebooks located in the notebooks/ directory.
If you wish to re-train the models, please ensure you have access to adequate computational resources. We used UNC's Longleaf services for this purpose. Pre-trained models can be found in the models/ directory for immediate use or evaluation.


