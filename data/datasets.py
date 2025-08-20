"""
Dataset loading and processing for quantum text generation
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import random
import re
from collections import Counter

from config.data_configs import get_dataset_info, get_sample_data, SAMPLE_DATA

class QuantumTextDataset(Dataset):
    """Base dataset class for quantum text generation"""
    
    def __init__(self, 
                 texts: List[str],
                 tokenizer: 'SimpleTokenizer',
                 max_length: int = 64,
                 pad_token: str = '<PAD>',
                 unk_token: str = '<UNK>'):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token = pad_token
        self.unk_token = unk_token
        
        # Tokenize all texts
        self.tokenized_texts = []
        for text in texts:
            tokens = self.tokenizer.encode(text)
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            self.tokenized_texts.append(tokens)
    
    def __len__(self):
        return len(self.tokenized_texts)
    
    def __getitem__(self, idx):
        tokens = self.tokenized_texts[idx]
        
        # Pad sequence
        if len(tokens) < self.max_length:
            pad_id = self.tokenizer.token_to_id(self.pad_token)
            tokens = tokens + [pad_id] * (self.max_length - len(tokens))
        
        # Create input and target sequences (for language modeling)
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'attention_mask': (input_ids != self.tokenizer.token_to_id(self.pad_token)).float()
        }

class SimpleTokenizer:
    """Simple tokenizer for small vocabulary datasets"""
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab = set()
        
        # Special tokens
        self.special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        for i, token in enumerate(self.special_tokens):
            self.word_to_id[token] = i
            self.id_to_word[i] = token
            self.vocab.add(token)
    
    def build_vocab(self, texts: List[str]):
        """Build vocabulary from texts"""
        # Tokenize and count words
        word_counts = Counter()
        for text in texts:
            words = self._tokenize_text(text)
            word_counts.update(words)
        
        # Add most frequent words to vocabulary
        most_common = word_counts.most_common(self.vocab_size - len(self.special_tokens))
        
        for word, _ in most_common:
            if word not in self.vocab:
                word_id = len(self.word_to_id)
                self.word_to_id[word] = word_id
                self.id_to_word[word_id] = word
                self.vocab.add(word)
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Simple text tokenization"""
        # Convert to lowercase and split by spaces/punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)  # Replace punctuation with spaces
        words = text.split()
        return words
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        words = self._tokenize_text(text)
        token_ids = [self.word_to_id.get('<BOS>', 2)]  # Start token
        
        for word in words:
            if word in self.word_to_id:
                token_ids.append(self.word_to_id[word])
            else:
                token_ids.append(self.word_to_id.get('<UNK>', 1))
        
        token_ids.append(self.word_to_id.get('<EOS>', 3))  # End token
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        words = []
        for token_id in token_ids:
            if token_id in self.id_to_word:
                word = self.id_to_word[token_id]
                if word not in ['<PAD>', '<BOS>', '<EOS>']:
                    words.append(word)
        return ' '.join(words)
    
    def token_to_id(self, token: str) -> int:
        """Get token ID"""
        return self.word_to_id.get(token, self.word_to_id.get('<UNK>', 1))
    
    def id_to_token(self, token_id: int) -> str:
        """Get token from ID"""
        return self.id_to_word.get(token_id, '<UNK>')
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.word_to_id)

class DatasetGenerator:
    """Generate synthetic datasets for quantum text generation"""
    
    @staticmethod
    def generate_simple_sentences(n_samples: int = 5000) -> List[str]:
        """Generate simple English sentences"""
        subjects = ['the cat', 'a dog', 'the bird', 'a child', 'the man', 'a woman', 
                   'the tree', 'a flower', 'the sun', 'the moon']
        verbs = ['runs', 'walks', 'flies', 'jumps', 'sits', 'stands', 'sleeps', 
                'plays', 'sings', 'dances']
        objects = ['in the park', 'on the table', 'under the tree', 'near the house',
                  'by the river', 'in the garden', 'on the hill', 'in the sky',
                  'through the forest', 'across the field']
        
        sentences = []
        for _ in range(n_samples):
            subject = random.choice(subjects)
            verb = random.choice(verbs)
            obj = random.choice(objects)
            sentence = f"{subject} {verb} {obj}."
            sentences.append(sentence)
        
        # Add some variety
        base_sentences = SAMPLE_DATA['simple_sentences'] * (n_samples // 100)
        sentences.extend(base_sentences[:n_samples - len(sentences)])
        
        return sentences[:n_samples]
    
    @staticmethod
    def generate_quantum_phrases(n_samples: int = 2000) -> List[str]:
        """Generate quantum computing related phrases"""
        base_phrases = SAMPLE_DATA['quantum_phrases']
        
        # Quantum terms
        quantum_terms = ['quantum', 'qubit', 'superposition', 'entanglement', 'coherence',
                        'decoherence', 'measurement', 'wave function', 'probability']
        actions = ['creates', 'destroys', 'measures', 'controls', 'manipulates',
                  'observes', 'influences', 'affects', 'determines']
        properties = ['states', 'systems', 'particles', 'information', 'data',
                     'circuits', 'gates', 'algorithms', 'computations']
        
        phrases = base_phrases.copy()
        
        while len(phrases) < n_samples:
            term = random.choice(quantum_terms)
            action = random.choice(actions)
            prop = random.choice(properties)
            phrase = f"{term.title()} {action} {prop} in quantum systems."
            phrases.append(phrase)
        
        return phrases[:n_samples]
    
    @staticmethod
    def expand_dataset(base_data: List[str], target_size: int) -> List[str]:
        """Expand a small dataset to target size with variations"""
        if len(base_data) >= target_size:
            return base_data[:target_size]
        
        expanded = base_data.copy()
        
        while len(expanded) < target_size:
            # Add original data multiple times with slight variations
            for text in base_data:
                if len(expanded) >= target_size:
                    break
                
                # Add original
                expanded.append(text)
                
                if len(expanded) >= target_size:
                    break
                
                # Add with minor variations (simple word replacement)
                varied = DatasetGenerator._create_variation(text)
                expanded.append(varied)
        
        return expanded[:target_size]
    
    @staticmethod
    def _create_variation(text: str) -> str:
        """Create a simple variation of the text"""
        # Simple word replacements
        replacements = {
            'the': 'a', 'a': 'the', 'big': 'large', 'small': 'tiny',
            'good': 'great', 'bad': 'poor', 'fast': 'quick', 'slow': 'gradual'
        }
        
        words = text.lower().split()
        for i, word in enumerate(words):
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in replacements and random.random() < 0.3:
                # Preserve punctuation
                punctuation = re.findall(r'[^\w]', word)
                new_word = replacements[clean_word]
                if punctuation:
                    new_word += ''.join(punctuation)
                words[i] = new_word
        
        return ' '.join(words)

def load_dataset(dataset_name: str, 
                split: str = 'train',
                max_samples: Optional[int] = None) -> List[str]:
    """Load a dataset by name"""
    if dataset_name not in SAMPLE_DATA:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Get base data
    if dataset_name == 'simple_sentences':
        if max_samples and max_samples > 100:
            texts = DatasetGenerator.generate_simple_sentences(max_samples)
        else:
            texts = SAMPLE_DATA[dataset_name]
    elif dataset_name == 'quantum_phrases':
        if max_samples and max_samples > 50:
            texts = DatasetGenerator.generate_quantum_phrases(max_samples)
        else:
            texts = SAMPLE_DATA[dataset_name]
    else:
        texts = SAMPLE_DATA[dataset_name]
        if max_samples:
            texts = DatasetGenerator.expand_dataset(texts, max_samples)
    
    # Split data
    if split == 'train':
        return texts[:int(0.8 * len(texts))]
    elif split == 'val':
        start_idx = int(0.8 * len(texts))
        end_idx = int(0.9 * len(texts))
        return texts[start_idx:end_idx]
    elif split == 'test':
        return texts[int(0.9 * len(texts)):]
    else:
        return texts

def create_dataloader(dataset_name: str,
                     split: str = 'train',
                     batch_size: int = 16,
                     max_length: int = 64,
                     vocab_size: int = 1000,
                     max_samples: Optional[int] = None,
                     shuffle: bool = True) -> Tuple[DataLoader, SimpleTokenizer]:
    """Create a DataLoader for the specified dataset"""
    
    # Load texts
    texts = load_dataset(dataset_name, split, max_samples)
    
    # Create tokenizer
    tokenizer = SimpleTokenizer(vocab_size)
    all_texts = []
    for split_name in ['train', 'val', 'test']:
        split_texts = load_dataset(dataset_name, split_name, max_samples)
        all_texts.extend(split_texts)
    
    tokenizer.build_vocab(all_texts)
    
    # Create dataset
    dataset = QuantumTextDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Keep simple for quantum computing
        pin_memory=False
    )
    
    return dataloader, tokenizer