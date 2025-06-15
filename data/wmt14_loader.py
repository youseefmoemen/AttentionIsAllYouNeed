from datasets import load_dataset
from torch.utils.data import Dataset
import re
import unicodedata
from typing import List, Tuple, Optional

class WMTDataset(Dataset):
    def __init__(self, split, min_length=5, max_length=512, max_length_ratio=3.0, 
                 remove_duplicates=True, normalize_unicode=True):
        super().__init__()
        self.split = split
        self.min_length = min_length
        self.max_length = max_length
        self.max_length_ratio = max_length_ratio
        self.remove_duplicates = remove_duplicates
        self.normalize_unicode = normalize_unicode
        
        # Load and preprocess data
        raw_data = load_dataset('wmt14', 'de-en', split=split)['translation']
        self.data = self._preprocess_data(raw_data)
        
        print(f"Dataset loaded: {len(raw_data)} -> {len(self.data)} samples after cleaning")

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text or not isinstance(text, str):
            return ""
            
        # Normalize unicode characters
        if self.normalize_unicode:
            text = unicodedata.normalize('NFKC', text)
        
        # Remove or replace problematic characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Fix common encoding issues
        text = text.replace('â€™', "'")
        text = text.replace('â€œ', '"')
        text = text.replace('â€�', '"')
        text = text.replace('â€"', '–')
        text = text.replace('â€"', '—')
        
        # Remove excessive punctuation
        text = re.sub(r'([.!?]){3,}', r'\1\1\1', text)
        text = re.sub(r'([,;:]){2,}', r'\1', text)
        
        text = re.sub(r'[''‛]', "'", text)    # Normalize various apostrophes to standard '
        text = re.sub(r'([.!?]){3,}', r'\1\1\1', text)  # Limit excessive punctuation

        return text

    def _is_valid_pair(self, source: str, target: str) -> bool:
        """Check if a translation pair is valid based on various criteria."""
        
        # Check for empty or None strings
        if not source or not target or not source.strip() or not target.strip():
            return False
            
        # Length filtering
        if len(source) < self.min_length or len(target) < self.min_length:
            return False
            
        if len(source) > self.max_length or len(target) > self.max_length:
            return False
            
        # Length ratio filtering (avoid very unbalanced pairs)
        length_ratio = max(len(source), len(target)) / min(len(source), len(target))
        if length_ratio > self.max_length_ratio:
            return False
            
        # Check for suspicious patterns
        
        # Too many repeated characters
        if re.search(r'(.)\1{10,}', source) or re.search(r'(.)\1{10,}', target):
            return False
            
        # Too many digits (likely not natural language)
        digit_ratio_source = len(re.findall(r'\d', source)) / len(source)
        digit_ratio_target = len(re.findall(r'\d', target)) / len(target)
        if digit_ratio_source > 0.3 or digit_ratio_target > 0.3:
            return False
            
        # Check for URLs, emails, or other non-linguistic content
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        
        if re.search(url_pattern, source) or re.search(url_pattern, target):
            return False
        if re.search(email_pattern, source) or re.search(email_pattern, target):
            return False
            
        # Check for excessive special characters
        special_char_ratio_source = len(re.findall(r'[^\w\s.,!?;:()\-\'"]', source)) / len(source)
        special_char_ratio_target = len(re.findall(r'[^\w\s.,!?;:()\-\'"]', target)) / len(target)
        if special_char_ratio_source > 0.2 or special_char_ratio_target > 0.2:
            return False
            
        # Language-specific checks
        
        # German should have some typical German characteristics
        german_chars = len(re.findall(r'[äöüßÄÖÜ]', source))
        if len(source) > 50 and german_chars == 0:
            # Long German text should have at least some umlauts
            pass  # Keep for now, but could be stricter
            
        # Check for identical source and target (shouldn't happen in translation)
        if source.lower().strip() == target.lower().strip():
            return False
            
        # Check for mostly punctuation
        word_count_source = len(re.findall(r'\b\w+\b', source))
        word_count_target = len(re.findall(r'\b\w+\b', target))
        if word_count_source < 2 or word_count_target < 2:
            return False
            
        return True

    def _preprocess_data(self, raw_data) -> List[Tuple[str, str]]:
        """Preprocess and clean the dataset."""
        cleaned_data = []
        seen_pairs = set() if self.remove_duplicates else None
        
        for item in raw_data:
            source = self._clean_text(item['de'])
            target = self._clean_text(item['en'])
            
            # Skip invalid pairs
            if not self._is_valid_pair(source, target):
                continue
                
            # Remove duplicates if requested
            if self.remove_duplicates:
                pair_key = (source.lower().strip(), target.lower().strip())
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
            
            cleaned_data.append((source, target))
        
        return cleaned_data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

    def get_statistics(self):
        """Get dataset statistics after preprocessing."""
        if not self.data:
            return {}
            
        source_lengths = [len(pair[0]) for pair in self.data]
        target_lengths = [len(pair[1]) for pair in self.data]
        
        return {
            'total_samples': len(self.data),
            'avg_source_length': sum(source_lengths) / len(source_lengths),
            'avg_target_length': sum(target_lengths) / len(target_lengths),
            'max_source_length': max(source_lengths),
            'max_target_length': max(target_lengths),
            'min_source_length': min(source_lengths),
            'min_target_length': min(target_lengths)
        }


def collate_fn(batch, tokenizer, max_seq_len):
    """Enhanced collate function with better error handling."""
    try:
        source, target = zip(*batch)
        
        # Additional cleaning at tokenization time
        source = [s.strip() for s in source if s and s.strip()]
        target = [t.strip() for t in target if t and t.strip()]
        
        if not source or not target:
            return None
            
        source_tokens, _, encoder_attention_mask = tokenizer(
            source,
            padding='max_length',
            max_length=max_seq_len,
            truncation=True,
            return_tensors=True
        )
        
        target_tokens, model_target, decoder_attention_mask = tokenizer(
            target,
            padding='max_length',
            max_length=max_seq_len, 
            truncation=True,
            return_tensors=True,
        )
        
        batch = {
            'encoder_input': {
                'source_tokens': source_tokens,
                'encoder_attention_mask': encoder_attention_mask
            },
            'decoder_input': {
                'target_tokens': target_tokens,
                'decoder_attention_mask': decoder_attention_mask
            },
            'model_target': model_target
        }
        return batch
        
    except Exception as e:
        print(f"Error in collate_fn: {e}")
        return None


# Example usage and testing
if __name__ == "__main__":
    # Load dataset with custom parameters
    dataset = WMTDataset(
        split='train[:100_000]', 
        min_length=256,      # Minimum 10 characters
        max_length=512,     # Maximum 400 characters  
        max_length_ratio=2.5,  # Max 2.5x length difference
        remove_duplicates=True,
        normalize_unicode=True
    )
    
    print("Dataset Statistics:")
    stats = dataset.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Sample a few examples
    print("\nSample translations:")
    for i in range(min(3, len(dataset))):
        source, target = dataset[i]
        print(f"DE: {source}")
        print(f"EN: {target}")
        print("-" * 50)