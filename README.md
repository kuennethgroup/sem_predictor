
# SEM Phase Analysis Web App

This Streamlit tool enables fast, ML-based analysis of BSE images from Cr-Si alloys developed as part of a Master's thesis. It detects different phases and elements, calculates error margins, and allows CSV data export.

## Features

- Multilingual interface (German/English)
- Analysis and visualization of phase fractions in BSE images
- Selection and comparison of different ML models (ResNet50)
- Upload of images (PNG, JPG, TIFF, BMP)
- Batch analysis of multiple images with error estimation
- Export of results as CSV
- User-friendly design with step-by-step instructions

## Installation

1. Clone the repository:

   git clone https://github.com/kuennethgroup/sem_predictor.git
   cd [path-to-repo-name]

2. Install required packages:

   pip install -r requirements.txt

3. Start the app:

   streamlit run app.py

## Basic Usage

1. Select model in the sidebar (two pretrained models available)
2. Choose the phases/elements to determine
3. Upload images (drag & drop or click)
4. Select images for analysis ("Select All" button is available)
5. Start analysis and view results in the table
6. Option to download results as CSV

## Structure

- `app.py`: Streamlit web app & ML workflow
- `models/`: Pretrained ResNet50 models
- `data/`: Scaling data/error values
- `requirements.txt`: Required Python packages

## Note

- Model paths and data are hardcoded in the script, please adjust to your own environment.
- Results include error estimates per phase/element according to model configuration.
- The app was specifically developed for a Master's thesis in Computational Materials Science.

---

**Contact:**  
Developed by Lukas as part of a Master's thesis. For questions, feel free to raise an issue in the repository!
