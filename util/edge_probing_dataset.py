import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


class EdgeProbingDataset:
    def __init__(self, mapping_file_path: str):
        """
        Initialize dataset creator with phoneme-to-feature mappings.
        
        Args:
            mapping_file_path: Path to mappings.json file
        """
        with open(mapping_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.phoneme_mappings = data['mappings']
        self.feature_names = data['features']
        self.num_features = len(self.feature_names)
        self.idx_to_feature = {idx: name for name, idx in self.feature_names.items()}
        
        print(f"Loaded {len(self.phoneme_mappings)} phonemes")
        print(f"Number of features: {self.num_features}")
        
    def get_binary_features(self, phoneme: str) -> List[int]:
        """
        This is just a function to convert phoneme to binary feature vector. so that it can 
        better visualize if a specific phoneme feature is present in the given sentences or not.
        
        Args:
            phoneme: IPA phoneme string (e.g., 'æ', 'ʈʂ')
            
        Returns:
            Binary list where 1 = feature present, 0 = absent/don't care
        """
        if phoneme not in self.phoneme_mappings:
            # Return all zeros for unknown phonemes
            return [0] * self.num_features
        
        raw_features = self.phoneme_mappings[phoneme]
        
        binary_features = [0] * self.num_features
        
        for feature_idx in raw_features:
            if feature_idx >= 0:
                binary_features[feature_idx] = 1
                
        return binary_features
    
    def get_feature_names_for_phoneme(self, phoneme: str) -> List[str]:
        """
        Get the names for a phoneme.
        """
        if phoneme not in self.phoneme_mappings:
            return []
        
        raw_features = self.phoneme_mappings[phoneme]
        present_features = []
        
        for feature_idx in raw_features:
            if feature_idx >= 0:
                present_features.append(self.idx_to_feature[feature_idx])
        
        return present_features
    
    def create_fixed_windows(self, 
                            sentences: List[List[str]], 
                            window_size: int = 5) -> List[Dict]:
        """
        Create fixed-size contextual windows for edge probing.
        For each phoneme in each sentence, create a fixed-size window of context and label it with
        the phonological features of the center (target) phoneme. 5 is a random number I picked.
        """
        if window_size % 2 == 0:
            raise ValueError("window_size must be odd")
        
        dataset = []
        pad_size = window_size // 2
        
        for sent_idx, sentence in enumerate(sentences):
            if len(sentence) < 1:
                continue
                
            # Pad with special token
            padded = ['<PAD>'] * pad_size + sentence + ['<PAD>'] * pad_size
            
            for pos in range(len(sentence)):
                target_phoneme = sentence[pos]
                
                # Skip if not in mapping
                if target_phoneme not in self.phoneme_mappings:
                    print(f"Warning: Phoneme '{target_phoneme}' not in mappings, skipping")
                    continue
                
                # Extract window centered on this position
                window = padded[pos:pos + window_size]
                
                # Get features for target phoneme
                features_binary = self.get_binary_features(target_phoneme)
                features_present = self.get_feature_names_for_phoneme(target_phoneme)
                
                dataset.append({
                    'sentence_id': sent_idx,
                    'position': pos,
                    'window': window,
                    'target_phoneme': target_phoneme,
                    'target_idx': pad_size,
                    'features_binary': features_binary,
                    'features_present': features_present,
                    'raw_feature_indices': self.phoneme_mappings[target_phoneme]
                })
        
        return dataset
    
    def create_span_windows(self,
                           sentences: List[List[str]],
                           min_span: int = 1,
                           max_span: int = 3) -> List[Dict]:
        """
        Create span-based windows for edge probing.
        Basically extracts spans of varying lengths and tests if the model can predict features from multi-phoneme sequences.
        """
        spans = []
        
        for sent_idx, sentence in enumerate(sentences):
            for span_len in range(min_span, max_span + 1):
                for start_idx in range(len(sentence) - span_len + 1):
                    end_idx = start_idx + span_len
                    span_tokens = sentence[start_idx:end_idx]
                    
                    # For feature labeling, use the central phoneme
                    if span_len == 1:
                        target_phoneme = span_tokens[0]
                    else:
                        # Use central phoneme for multi-token spans
                        central_idx = span_len // 2
                        target_phoneme = span_tokens[central_idx]
                    
                    # Skip if phoneme not in mapping
                    if target_phoneme not in self.phoneme_mappings:
                        continue
                    
                    features_binary = self.get_binary_features(target_phoneme)
                    features_present = self.get_feature_names_for_phoneme(target_phoneme)
                    
                    spans.append({
                        'sentence_id': sent_idx,
                        'span': span_tokens,
                        'start': start_idx,
                        'end': end_idx,
                        'span_length': span_len,
                        'target_phoneme': target_phoneme,
                        'features_binary': features_binary,
                        'features_present': features_present
                    })
        
        return spans
    
    def create_weighted_windows(self,
                               sentences: List[List[str]],
                               window_size: int = 7) -> List[Dict]:
        """
        Create windows with position-aware weighting.
        Uses distance-based weights to emphasize closer context when
        predicting features.
        """
        if window_size % 2 == 0:
            raise ValueError("window_size must be odd")
        
        dataset = []
        pad_size = window_size // 2
        
        # Create Gaussian-like weights (center = 1.0, edges lower)
        weights = []
        for i in range(window_size):
            distance = abs(i - pad_size)
            weight = np.exp(-distance / 2.0)
            weights.append(weight)
        
        for sent_idx, sentence in enumerate(sentences):
            if len(sentence) < 1:
                continue
                
            padded = ['<PAD>'] * pad_size + sentence + ['<PAD>'] * pad_size
            
            for pos in range(len(sentence)):
                target_phoneme = sentence[pos]
                
                if target_phoneme not in self.phoneme_mappings:
                    continue
                
                window = padded[pos:pos + window_size]
                features_binary = self.get_binary_features(target_phoneme)
                features_present = self.get_feature_names_for_phoneme(target_phoneme)
                
                dataset.append({
                    'sentence_id': sent_idx,
                    'position': pos,
                    'window': window,
                    'weights': weights,
                    'target_idx': pad_size,
                    'target_phoneme': target_phoneme,
                    'features_binary': features_binary,
                    'features_present': features_present
                })
        
        return dataset
    
    def get_statistics(self, dataset: List[Dict]) -> Dict:
        """
        Compute statistics about the dataset.
        """
        if not dataset:
            return {"error": "Empty dataset"}
        
        phoneme_counts = defaultdict(int)
        for sample in dataset:
            phoneme_counts[sample['target_phoneme']] += 1
        feature_counts = defaultdict(int)
        for sample in dataset:
            for feature in sample['features_present']:
                feature_counts[feature] += 1
        feature_vectors = [sample['features_binary'] for sample in dataset]
        avg_features_per_phoneme = np.mean([sum(fv) for fv in feature_vectors])
        
        return {
            'total_samples': len(dataset),
            'unique_phonemes': len(phoneme_counts),
            'unique_sentences': len(set(s['sentence_id'] for s in dataset)),
            'avg_features_per_phoneme': avg_features_per_phoneme,
            'most_common_phonemes': sorted(phoneme_counts.items(), 
                                          key=lambda x: x[1], 
                                          reverse=True)[:10],
            'most_common_features': sorted(feature_counts.items(),
                                          key=lambda x: x[1],
                                          reverse=True)[:10]
        }
    
    def save_dataset(self, dataset: List[Dict], output_path: str):
        """Save dataset to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(dataset)} samples to {output_path}")
    
    def load_dataset(self, input_path: str) -> List[Dict]:
        """Load dataset from JSON file."""
        with open(input_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        print(f"Loaded {len(dataset)} samples from {input_path}")
        return dataset


# Example usage and testing, I asked for GPT to came up this part, I have not rlly test on it
# but im working on a testing dataset or demon script to test if the above functions work as intended.
if __name__ == "__main__":
    print("=" * 80)
    print("Edge Probing Dataset Creator - Example Usage")
    print("=" * 80)
    
    # Example: Create dataset from sample sentences
    # Note: In real use, you'd load sentences from your actual corpus
    
    sample_sentences = [
        ['k', 'æ', 't'],      # "cat"
        ['d', 'ɔ', 'ɡ'],      # "dog"  
        ['h', 'ə', 'l', 'o'],  # "hello"
        ['w', 'ɜ', 'ɹ', 'l', 'd']  # "world"
    ]
    
    print("\nSample sentences:")
    for i, sent in enumerate(sample_sentences):
        print(f"  {i}: {' '.join(sent)}")
    
    # This would be run with actual path in practice
    # creator = EdgeProbingDataset('/mnt/user-data/uploads/mappings.json')
    print("\n[In actual use, initialize with: EdgeProbingDataset('path/to/mappings.json')]")
    print("\nExample dataset structures generated by each alternative:")
    print("\n1. FIXED WINDOWS (RECOMMENDED)")
    print("   - For each phoneme, extracts fixed-size context window")
    print("   - Standard approach in edge probing literature")
    print("   - Window example: ['<PAD>', '<PAD>', 'k', 'æ', 't']")
    print("   - Target: 'k' (position 2)")
    
    print("\n2. SPAN WINDOWS")
    print("   - Extracts variable-length spans")
    print("   - Tests multi-phoneme feature encoding")
    print("   - Span examples: ['k'], ['k', 'æ'], ['k', 'æ', 't']")
    
    print("\n3. WEIGHTED WINDOWS")
    print("   - Like fixed windows but with position weights")
    print("   - Emphasizes closer context")
    print("   - Weights: [0.14, 0.36, 0.61, 1.0, 0.61, 0.36, 0.14]")
    
    print("\n" + "=" * 80)