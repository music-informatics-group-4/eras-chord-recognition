# Eras Chord Recognition

HMM-based chord recognition project to compare performance across different dataset subsets (eras, genres, etc.).

## Project Structure

```
eras-chord-recognition/
├── data/                    # Dataset files
├── models/                  # Saved trained models (created during training)
├── results/                 # Training and evaluation outputs
├── hmm_model.py            # HMM class definition (skeleton)
├── data_utils.py           # Data loading and preprocessing utilities
├── train.py                # Training pipeline script
├── evaluate.py             # Model evaluation and comparison
└── environment.yml         # Conda environment specification
```

## Environment Setup

```bash
conda env create -f environment.yml
conda activate eras-chord-recognition
```
## Working with the Dataset

Download the data from the [CrossEra Dataset](https://www.audiolabs-erlangen.de/resources/MIR/cross-era). You need:
- Annotations
- Chroma features 
- Chord features

Organize your data directory as follows:

```
data/
├── cross-era_annotations.csv
├── chroma_features/
│   ├── file1.csv
│   ├── file2.csv
│   └── ...
└── chords/
    ├── file1.csv
    ├── file2.csv
    └── ...
```

## Data Loading

Test the data loading:
```bash
python data_utils.py
```

This will:
- Load and validate your dataset
- Create cached `.pkl` files for faster subsequent loads
- Show some information on what was loaded

## Plan for implementation

1. We set up our environments
2. Download the dataset into data/ directory
3. Implement the skeleton methods in each module
4. Run training: `python train.py`
5. Run evaluation: `python evaluate.py`

