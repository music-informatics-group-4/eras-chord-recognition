# training script for HMM chord recognition model

import os
import argparse
from typing import Dict, List
import numpy as np
from data_utils import load_data, create_subsets_by_era, MusicPiece
from hmm_model import ChordHMM


def parse_args():
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(description='Train HMM models for chord recognition')
    
    parser.add_argument('--subset-by', type=str, default='all',
                       choices=['era', 'all'],
                       help='How to split the dataset (default: all)')

    parser.add_argument('--output-dir', type=str, default='models',
                       help='Directory to save trained models (default: models/)')
    
    parser.add_argument('--no-cache', action='store_true',
                       help='Ignore cached data and reload from source')
    
    return parser.parse_args()


def train_hmm_for_subset(subset_name: str, 
                        music_pieces: List[MusicPiece],
                        output_dir: str) -> ChordHMM:
    """
    Train an HMM model for a specific subset of data.
    """
    print(f"\nTraining HMM for subset: {subset_name}")
    print(f"{'='*60}")

    # extracting chroma and chord sequences
    # the shape is (List[List[Seqlength, 12]]) and (List[List[Seqlength]])
    chroma_sequences = [piece.chroma_features for piece in music_pieces]
    chord_sequences = [piece.chord_labels for piece in music_pieces]

    # initialize the HMM model
    model = ChordHMM()
    
    model.train(chroma_sequences, chord_sequences)
    
    # Save the trained model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"hmm_{subset_name}.pkl")
    
    print(f"Saving model to {model_path}...")
    
    model.save(model_path)
    
    print(f"Training complete for {subset_name}")
    
    return model


def train_all_subsets(subsets: Dict[str, List[MusicPiece]], 
                     output_dir: str):
    """
    Train HMM models for all subsets of data.
    """   
    trained_models = {}
    
    for subset_name, pieces in subsets.items():
        model = train_hmm_for_subset(subset_name, pieces, output_dir)
        trained_models[subset_name] = model
    
    return trained_models


def main():
    """
    Main training pipeline.
    
    Workflow:
    1. Load data from CrossEra dataset (with caching for speed)
    2. Split data into subsets (by era or train on all)
    3. Train separate HMM models for each subset
    4. Save trained models to disk for evaluation
    """
    args = parse_args()
    
    print("="*60)
    print("HMM Chord Recognition - Training Pipeline")
    print("="*60)
    
    # Step 1: Load data
    music_pieces = load_data(use_cache=not args.no_cache)

    # subsample music_pieces for quick testing
    # TODO: Remove or adjust for full training
    music_pieces = music_pieces[:10]  # Remove or adjust for full training
    
    if not music_pieces:
        print("Error: No data loaded. Please check your data directory.")
        return
    
    # Step 2: Create subsets
    if args.subset_by == 'era':
        subsets = create_subsets_by_era(music_pieces)
    else:  # 'all'
        subsets = {'all': music_pieces}
    
    # Step 3 & 4: Train models for each subset, saving to disk
    trained_models = train_all_subsets(
        subsets, 
        args.output_dir
    )

if __name__ == "__main__":
    main()