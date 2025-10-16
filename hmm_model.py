# Hidden Markov Model for Chord Recognition

import numpy as np
import pickle
from typing import List, Tuple, Optional
from hmmlearn import hmm


class ChordHMM:
    """
    Hidden Markov Model for chord recognition from chroma features.
    """
    
    def __init__(self):
        """Initialize empty model."""
        self.model = None  # hmmlearn.hmm.GaussianHMM
        self.chord_labels = None  # List[str] - ordered list of unique chords
        self.chord_to_idx = None  # Dict[str, int] - chord name -> state index
        self.idx_to_chord = None  # Dict[int, str] - state index -> chord name
        
    def _build_chord_vocabulary(self, chord_sequences: List[List[str]]) -> List[str]:
        """
        Build fixed chord vocabulary.
        """
        chroma = ["A", "Bb", "B", "C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab"]
        
        # all major and minor chords + "N"
        chords = ["N"]
        for c in chroma:
            chords.append(c + "maj")
        for c in chroma:
            chords.append(c + "min")
        
        return chords
    

    def _simplify_chord_label(self, chord: str) -> str:
        """
        Simplify chord labels to match the fixed vocabulary.
        """
        # applying simplification rules from notebook
        chord = chord.replace("dim", "min")
        chord = chord.replace("aug", "maj")
        chord = ''.join(chord.split("_")[:2])
        return chord
    
    def _simplify_chord_sequences(self, chord_sequences: List[List[str]]) -> List[List[str]]:
        """
        Simplify all chord sequences to match the fixed vocabulary.
        """
        simplified_sequences = []
        
        for chord_seq in chord_sequences:
            simplified_seq = [self._simplify_chord_label(chord) for chord in chord_seq]
            simplified_sequences.append(simplified_seq)
        
        return simplified_sequences

    def _compute_empirical_transitions(self, 
                                      chord_sequences: List[List[str]]) -> np.ndarray:
        """
        Compute empirical transition matrix from training sequences.
        Based on notebook's empirical_transition_matrix() function.
        """
        n_chords = len(self.chord_labels)
        transition_matrix = np.zeros((n_chords, n_chords))
        
        # counting transitions
        for chord_seq in chord_sequences:
            for i in range(len(chord_seq) - 1):
                current_chord = chord_seq[i]
                next_chord = chord_seq[i + 1]
                
                # if chord isn't in vocabulary, skip
                if current_chord not in self.chord_to_idx or next_chord not in self.chord_to_idx:
                    continue
                
                current_idx = self.chord_to_idx[current_chord]
                next_idx = self.chord_to_idx[next_chord]
                
                # self-transition vs to next chord
                if current_chord == next_chord:
                    transition_matrix[current_idx, current_idx] += 1
                else:
                    transition_matrix[current_idx, next_idx] += 1
        
        # avoid non-zero/Nan probabilities
        transition_matrix += 1e-10
        
        # normalizing each row to sum to 1
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = transition_matrix / row_sums
        
        return transition_matrix
    
    def _initialize_emission_means(self, chroma_data: np.ndarray) -> np.ndarray:
        """
        Initialize emission means based on music theory chord templates.
        """
        chroma_notes = ["A", "Bb", "B", "C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab"]
        
        chord_note_names = {
            'Amaj': ['A', 'C#', 'E'],
            'Bbmaj': ['Bb', 'D', 'F'],
            'Bmaj': ['B', 'Eb', 'F#'],
            'Cmaj': ['C', 'E', 'G'],
            'C#maj': ['C#', 'F', 'Ab'],
            'Dmaj': ['D', 'F#', 'A'],
            'Ebmaj': ['Eb', 'G', 'Bb'],
            'Emaj': ['E', 'Ab', 'B'],
            'Fmaj': ['F', 'A', 'C'],
            'F#maj': ['F#', 'Bb', 'C#'],
            'Gmaj': ['G', 'B', 'D'],
            'Abmaj': ['Ab', 'C', 'Eb'],
            'Amin': ['A', 'C', 'E'],
            'Bbmin': ['Bb', 'C#', 'F'],
            'Bmin': ['B', 'D', 'F#'],
            'Cmin': ['C', 'Eb', 'G'],
            'C#min': ['C#', 'E', 'Ab'],
            'Dmin': ['D', 'F', 'A'],
            'Ebmin': ['Eb', 'F#', 'Bb'],
            'Emin': ['E', 'G', 'B'],
            'Fmin': ['F', 'Ab', 'C'],
            'F#min': ['F#', 'A', 'C#'],
            'Gmin': ['G', 'Bb', 'D'],
            'Abmin': ['Ab', 'B', 'Eb'],
        }
        
        # we get the mean chroma value across all data
        chroma_mean_value = chroma_data.mean()

        means = [
            np.ones(12) * 0.01
        ]
        
        for chord in self.chord_labels[1:]:
            chord_mean = np.zeros(12)
            
            if chord in chord_note_names:
                # initialize chord tones to chroma mean value
                for note_name in chord_note_names[chord]:
                    note_idx = chroma_notes.index(note_name)
                    chord_mean[note_idx] = chroma_mean_value
            
            means.append(chord_mean)
        
        return np.array(means)
        
    def train(self, chroma_sequences: List[np.ndarray], chord_sequences: List[List[str]]):
        """
        Train HMM on chroma features and chord labels.
        """
        # building the chord vocabulary and mapping chords to numerical indices
        self.chord_labels = self._build_chord_vocabulary(chord_sequences)
        self.chord_to_idx = {chord: i for i, chord in enumerate(self.chord_labels)}
        self.idx_to_chord = {i: chord for i, chord in enumerate(self.chord_labels)}
        
        n_chords = len(self.chord_labels)

        # simplifying
        ## TODO: do we need to simplify here? what chords do we want to include?
        simplified_sequences = self._simplify_chord_sequences(chord_sequences)

        # preparing the data for the HMM training, making the same as in notebook
        X_train = np.vstack(chroma_sequences)
        lengths = [len(seq) for seq in chroma_sequences]
        
        # compute the empirical transition matrix, and initialize emissions
        transition_matrix = self._compute_empirical_transitions(simplified_sequences)
        emission_means = self._initialize_emission_means(X_train)
        
        # variances are set to overall std
        ## TODO: do we want to have different variances for different chords?
        chroma_std = X_train.std()
        emission_covars = np.ones((n_chords, 12)) * chroma_std
        
        # initializing and training
        print("Training HMM...")
        np.random.seed(42)
        self.model = hmm.GaussianHMM(
            n_components=n_chords,
            covariance_type="diag",
            n_iter=1,  ## TODO: increase number of iterations
            params="s",
            init_params="s"
        )
        
        # setting initial params
        #self.model.startprob_ = np.ones(n_chords) / n_chords  # Uniform start probabilities
        self.model.transmat_ = transition_matrix
        self.model.means_ = emission_means
        self.model.covars_ = emission_covars
        
        # actual training
        self.model.fit(X_train, lengths=lengths)

    def predict(self, chroma_features: np.ndarray) -> List[str]:
        """
        Predict chord sequence from chroma features using Viterbi algorithm.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # predict the state sequence
        predicted_states = self.model.predict(chroma_features)
        
        # convert the indices back to chord labels
        chord_predictions = [self.idx_to_chord[idx] for idx in predicted_states]
        
        return chord_predictions

    def save(self, file_path: str):
        """
        Save model to disk using pickle.
        """
        model_data = {
            'model': self.model,
            'chord_labels': self.chord_labels,
            'chord_to_idx': self.chord_to_idx,
            'idx_to_chord': self.idx_to_chord,
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {file_path}")

    def load(self, file_path: str):
        """
        Load model from disk.
        """
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.chord_labels = model_data['chord_labels']
        self.chord_to_idx = model_data['chord_to_idx']
        self.idx_to_chord = model_data['idx_to_chord']
        
        print(f"Model loaded from {file_path}")