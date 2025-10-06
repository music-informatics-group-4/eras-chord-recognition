# data utilities for working with the cross-era dataset

import pandas as pd
import numpy as np
import os
import glob
import pickle
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class MusicPiece:
    """
    Data structure for a single piece of music with all its information.
    """
    filename: str
    cross_era_id: str
    class_name: str
    instrumentation: str
    key: str
    mode: str
    composer: str
    comp_lifetime: str
    country: str
    chroma_features: np.ndarray  # Shape: (n_frames, 12)
    chord_labels: List[str]      # Length: n_frames
    timestamps: np.ndarray       # Shape: (n_frames,)


def load_annotations(annotations_path: str = "data/cross-era_annotations.csv") -> pd.DataFrame:
    """
    Load the cross-era annotations CSV file.
    """
    if not os.path.exists(annotations_path):
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")
    
    annotations = pd.read_csv(annotations_path)
    
    # Validate required columns
    required_cols = ['Class', 'Filename', 'CrossEra-ID', 'Instrumentation', 
                     'Key', 'Mode', 'Composer', 'CompLifetime', 'Country']
    missing_cols = [col for col in required_cols if col not in annotations.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in annotations: {missing_cols}")
    
    return annotations


def load_chroma_features(chroma_dir: str = "data/chroma_features") -> Dict[str, pd.DataFrame]:
    """
    Load all chroma feature CSV files from the chroma_features directory.
    """
    if not os.path.exists(chroma_dir):
        raise FileNotFoundError(f"Chroma features directory not found: {chroma_dir}")
    
    chroma_files = glob.glob(os.path.join(chroma_dir, "*.csv"))
    if not chroma_files:
        raise ValueError(f"No CSV files found in {chroma_dir}")
    
    chroma_data = {}
    
    for file_path in chroma_files:
        csv_filename = os.path.basename(file_path)
        df = pd.read_csv(file_path, header=None)
        
        # Validate structure: should have Class/Filename, Time, and 12 chroma columns
        if len(df.columns) < 14:  # filename + time + 12 chroma features
            print(f"Warning: {csv_filename} has fewer columns than expected ({len(df.columns)})")
            continue
        
        # Process the dataframe - group by detecting filename changes
        current_filename = None
        current_group_rows = []
        
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            # Check if first column has a new filename (non-empty)
            first_col_value = row.iloc[0]
            
            if pd.notna(first_col_value) and str(first_col_value).strip():  # New filename detected
                # Save previous group if it exists
                if current_filename and current_group_rows:
                    # Extract just the filename part (after the class/)
                    if '/' in current_filename:
                        individual_filename = current_filename.split('/', 1)[1]
                    else:
                        individual_filename = current_filename
                    
                    # Create DataFrame from collected rows (time + chroma features)
                    group_df = pd.DataFrame(current_group_rows)
                    chroma_data[individual_filename] = group_df
                
                # Start new group
                current_filename = str(first_col_value)
                current_group_rows = []
            
            # Add current row's time and chroma data to the group
            if current_filename:  # Only add if we have a filename
                row_data = row.iloc[1:].values  # Skip first column, take time + chroma
                current_group_rows.append(row_data)
        
        # Don't forget the last group
        if current_filename and current_group_rows:
            if '/' in current_filename:
                individual_filename = current_filename.split('/', 1)[1]
            else:
                individual_filename = current_filename
            
            group_df = pd.DataFrame(current_group_rows)
            chroma_data[individual_filename] = group_df

    
    return chroma_data


def load_chord_annotations(chords_dir: str = "data/chords") -> Dict[str, pd.DataFrame]:
    """
    Load all chord annotation CSV files from the chords directory.
    """
    if not os.path.exists(chords_dir):
        raise FileNotFoundError(f"Chord annotations directory not found: {chords_dir}")
    
    chord_files = glob.glob(os.path.join(chords_dir, "*.csv"))
    if not chord_files:
        raise ValueError(f"No CSV files found in {chords_dir}")
    
    chord_data = {}
    
    for file_path in chord_files:
        csv_filename = os.path.basename(file_path)
        df = pd.read_csv(file_path, header=None)
        
        # Validate structure: should have Class/filename, Time, Chord
        if len(df.columns) < 3:
            print(f"Warning: {csv_filename} has fewer columns than expected ({len(df.columns)})")
            continue
        
        # Process the dataframe - group by detecting filename changes
        current_filename = None
        current_group_rows = []

        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            # Check if first column has a new filename (non-empty)
            first_col_value = row.iloc[0]
            
            if pd.notna(first_col_value) and str(first_col_value).strip():  # New filename detected
                # Save previous group if it exists
                if current_filename and current_group_rows:
                    # Extract just the filename part (after the class/)
                    if '/' in current_filename:
                        individual_filename = current_filename.split('/', 1)[1]
                    else:
                        individual_filename = current_filename
                    
                    # Create DataFrame from collected rows (time + chord)
                    group_df = pd.DataFrame(current_group_rows)
                    chord_data[individual_filename] = group_df
                
                # Start new group
                current_filename = str(first_col_value)
                current_group_rows = []
            
            # Add current row's time and chord data to the group
            if current_filename:  # Only add if we have a filename
                row_data = row.iloc[1:].values  # Skip first column, take time + chord
                current_group_rows.append(row_data)
        
        # Don't forget the last group
        if current_filename and current_group_rows:
            if '/' in current_filename:
                individual_filename = current_filename.split('/', 1)[1]
            else:
                individual_filename = current_filename
            
            group_df = pd.DataFrame(current_group_rows)
            chord_data[individual_filename] = group_df
    
    return chord_data


def match_filename_to_data(annotations: pd.DataFrame, 
                          chroma_data: Dict[str, pd.DataFrame],
                          chord_data: Dict[str, pd.DataFrame]) -> Dict[str, str]:
    """
    Create mapping between annotation filenames and data filenames.
    """
    filename_mapping = {}
    
    failed_matches = []
    
    for _, row in annotations.iterrows():
        annotation_filename = row['Filename']
        
        # Try to find matching files in chroma and chord data
        chroma_match = None
        chord_match = None
        
        # Look for exact matches
        if annotation_filename in chroma_data:
            chroma_match = annotation_filename
        if annotation_filename in chord_data:
            chord_match = annotation_filename
        
        if chroma_match and chord_match:
            filename_mapping[annotation_filename] = {
                'chroma_file': chroma_match,
                'chord_file': chord_match
            }
        else:
            failed_matches.append({
                'filename': annotation_filename,
                'chroma_match': chroma_match,
                'chord_match': chord_match
            })

    for failed in failed_matches:
        print(f"Warning: Could not find matching data files for {failed['filename']}")
    
    return filename_mapping


def align_chroma_and_chords(chroma_df: pd.DataFrame, 
                           chord_df: pd.DataFrame) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Align chroma features with chord labels based on timestamps.
    """

    # Get time columns (assume first column is time after filename removal)
    chroma_times = chroma_df.iloc[:, 0].values  # Changed from 1 to 0
    chord_times = chord_df.iloc[:, 0].values    # Changed from 1 to 0
    
    chroma_features = chroma_df.iloc[:, 1:13].values  # Changed from 2:14 to 1:13
    
    chord_labels = chord_df.iloc[:, 1].values  # Changed from 2 to 1
    
    aligned_chroma = []
    aligned_chords = []
    aligned_times = []

    # iterating over all chroma_times and adding the chord if it exists, if not add a "No Chord" label
    chord_time_to_label = {time: label for time, label in zip(chord_times, chord_labels)}
    for i, chroma_time in enumerate(chroma_times):
        if chroma_time in chord_time_to_label:
            aligned_chroma.append(chroma_features[i])
            aligned_chords.append(chord_time_to_label[chroma_time])
            aligned_times.append(chroma_time)
        else:
            aligned_chroma.append(chroma_features[i])
            aligned_chords.append("N")  # No Chord label
            aligned_times.append(chroma_time)

    return (np.array(aligned_chroma), aligned_chords, np.array(aligned_times))


def save_processed_data(chroma_data: Dict[str, pd.DataFrame], 
                       chord_data: Dict[str, pd.DataFrame],
                       cache_dir: str = "data/cache") -> None:
    """
    Save processed chroma and chord data to cache files for faster loading.
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"Saving processed data to {cache_dir}...")
    
    with open(os.path.join(cache_dir, "chroma_data.pkl"), "wb") as f:
        pickle.dump(chroma_data, f)
    
    with open(os.path.join(cache_dir, "chord_data.pkl"), "wb") as f:
        pickle.dump(chord_data, f)
    
    print("Processed data saved to cache!")


def load_processed_data(cache_dir: str = "data/cache") -> Tuple[Optional[Dict[str, pd.DataFrame]], Optional[Dict[str, pd.DataFrame]]]:
    """
    Load processed chroma and chord data from cache files.
    """
    chroma_cache = os.path.join(cache_dir, "chroma_data.pkl")
    chord_cache = os.path.join(cache_dir, "chord_data.pkl")
    
    if not (os.path.exists(chroma_cache) and os.path.exists(chord_cache)):
        return None, None
    
    try:
        print(f"Loading processed data from {cache_dir}...")
        
        with open(chroma_cache, "rb") as f:
            chroma_data = pickle.load(f)
        
        with open(chord_cache, "rb") as f:
            chord_data = pickle.load(f)
        
        print(f"Loaded {len(chroma_data)} chroma files and {len(chord_data)} chord files from cache")
        return chroma_data, chord_data
        
    except Exception as e:
        print(f"Error loading cache: {e}")
        return None, None


def check_cache_validity(cache_dir: str = "data/cache",
                        chroma_dir: str = "data/chroma_features",
                        chords_dir: str = "data/chords") -> bool:
    """
    Check if cache files exist, and are newer than source data files.
    """
    chroma_cache = os.path.join(cache_dir, "chroma_data.pkl")
    chord_cache = os.path.join(cache_dir, "chord_data.pkl")
    
    if not (os.path.exists(chroma_cache) and os.path.exists(chord_cache)):
        return False
    
    # Get cache modification times
    cache_time = min(os.path.getmtime(chroma_cache), os.path.getmtime(chord_cache))
    
    # Check if any source files are newer than cache
    for source_dir in [chroma_dir, chords_dir]:
        if os.path.exists(source_dir):
            for file_path in glob.glob(os.path.join(source_dir, "*.csv")):
                if os.path.getmtime(file_path) > cache_time:
                    return False
    
    return True


def load_data(use_cache: bool = True) -> List[MusicPiece]:
    """
    Load and combine all data sources into a list of MusicPiece objects.
    """
    print("Loading annotations...")
    annotations = load_annotations()
    
    # Try to load from cache if requested and valid
    chroma_data = None
    chord_data = None
    
    if use_cache and check_cache_validity():
        chroma_data, chord_data = load_processed_data()
    
    # If cache loading failed or not requested, process from scratch
    if chroma_data is None or chord_data is None:
        print("Loading chroma features...")
        chroma_data = load_chroma_features()
        
        print("Loading chord annotations...")
        chord_data = load_chord_annotations()
        
        # Save to cache for next time
        if use_cache:
            save_processed_data(chroma_data, chord_data)

    filename_mapping = match_filename_to_data(annotations, chroma_data, chord_data)
    
    music_pieces = []
    
    print("Aligning data...")
    for _, row in annotations.iterrows():
        filename = row['Filename']
        
        if filename not in filename_mapping:
            continue
        
        chroma_file = filename_mapping[filename]['chroma_file']
        chord_file = filename_mapping[filename]['chord_file']
        
        # Align chroma and chord data
        chroma_features, chord_labels, timestamps = align_chroma_and_chords(
            chroma_data[chroma_file], 
            chord_data[chord_file]
        )

        if len(chroma_features) == 0:
            print(f"Warning: No aligned data for {filename}")
            continue
        
        # Create MusicPiece object
        piece = MusicPiece(
            filename=filename,
            cross_era_id=row['CrossEra-ID'],
            class_name=row['Class'],
            instrumentation=row['Instrumentation'],
            key=row['Key'],
            mode=row['Mode'],
            composer=row['Composer'],
            comp_lifetime=row['CompLifetime'],
            country=row['Country'],
            chroma_features=chroma_features,
            chord_labels=chord_labels,
            timestamps=timestamps
        )
        
        music_pieces.append(piece)
    
    print(f"Successfully loaded {len(music_pieces)} pieces")
    return music_pieces


def create_subsets_by_era(music_pieces: List[MusicPiece]) -> Dict[str, List[MusicPiece]]:
    """
    Create subsets of music pieces grouped by era (Class).
    """
    subsets = {}
    
    for piece in music_pieces:
        era = piece.class_name
        if era not in subsets:
            subsets[era] = []
        subsets[era].append(piece)
    
    return subsets


def validate_data():
    """
    Validate that our dataset is complete and correctly formatted.
    """
    try:
        music_pieces = load_data()
        
        print(f"Validation successful! Loaded {len(music_pieces)} pieces")
        
        # Print some statistics
        eras = set(piece.class_name for piece in music_pieces)
        print(f"Eras found: {sorted(eras)}")
        
        for era in sorted(eras):
            era_pieces = [p for p in music_pieces if p.class_name == era]
            total_frames = sum(len(p.chord_labels) for p in era_pieces)
            print(f"  {era}: {len(era_pieces)} pieces, {total_frames} total frames")
        
        return True
        
    except Exception as e:
        print(f"Validation failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing data loading...")
    print("=" * 50)
    
    # Run validation
    success = validate_data()
    
    if success:
        print("\n" + "=" * 50)
        print("✅ Data loading test PASSED!")
        print("Your data files are properly formatted and loading correctly.")
    else:
        print("\n" + "=" * 50)
        print("❌ Data loading test FAILED!")
        print("Please check your data files and file paths.")