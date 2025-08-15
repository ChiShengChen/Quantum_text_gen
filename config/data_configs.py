"""
Dataset configurations for quantum text generation
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import os

@dataclass
class DatasetInfo:
    """Information about a dataset"""
    name: str
    description: str
    size: int
    vocab_size: int
    avg_length: float
    max_length: int
    source: str

# Small datasets suitable for quantum computing
QUANTUM_DATASETS = {
    'simple_sentences': DatasetInfo(
        name='simple_sentences',
        description='Simple English sentences for basic language modeling',
        size=5000,
        vocab_size=500,
        avg_length=8.5,
        max_length=20,
        source='generated'
    ),
    
    'short_stories': DatasetInfo(
        name='short_stories',
        description='Very short story fragments (1-2 sentences)',
        size=3000,
        vocab_size=800,
        avg_length=12.3,
        max_length=30,
        source='generated'
    ),
    
    'quantum_phrases': DatasetInfo(
        name='quantum_phrases',
        description='Physics and quantum computing related phrases',
        size=2000,
        vocab_size=600,
        avg_length=9.8,
        max_length=25,
        source='curated'
    ),
    
    'haiku': DatasetInfo(
        name='haiku',
        description='Traditional Japanese haiku poems (3 lines)',
        size=1500,
        vocab_size=400,
        avg_length=11.2,
        max_length=35,
        source='public_domain'
    ),
    
    'proverbs': DatasetInfo(
        name='proverbs',
        description='Short proverbs and sayings',
        size=2500,
        vocab_size=600,
        avg_length=7.8,
        max_length=20,
        source='public_domain'
    )
}

# Sample data for each dataset type
SAMPLE_DATA = {
    'simple_sentences': [
        "The cat sits on the mat.",
        "Birds fly in the sky.",
        "Water flows down the river.",
        "The sun shines bright today.",
        "Children play in the park.",
        "Books contain many stories.",
        "Music fills the quiet room.",
        "Flowers bloom in spring time.",
        "Dogs run fast through fields.",
        "Stars twinkle at night."
    ],
    
    'short_stories': [
        "Once upon a time, there lived a wise old owl.",
        "The magic key opened a door to another world.",
        "She found a message in a bottle on the beach.",
        "The clock struck midnight as the spell broke.",
        "A small dragon lived in the garden shed.",
        "The lighthouse keeper saw strange lights at sea.",
        "The old book whispered secrets to those who listened.",
        "Rain began to fall on the empty street.",
        "The treasure map led to an unexpected discovery.",
        "A gentle breeze carried the scent of roses."
    ],
    
    'quantum_phrases': [
        "Quantum entanglement connects distant particles.",
        "The wave function collapses upon measurement.",
        "Superposition allows multiple states simultaneously.",
        "Quantum computing uses qubits instead of bits.",
        "Uncertainty principle limits precise knowledge.",
        "Quantum tunneling enables barrier penetration.",
        "Coherence time determines quantum state stability.",
        "Bell inequalities test local realism.",
        "Quantum gates manipulate qubit states.",
        "Decoherence destroys quantum information."
    ],
    
    'haiku': [
        "Ancient pond ripples\nFrog jumps into silent depths\nWater sounds echo",
        "Cherry blossoms fall\nPetals dance on morning breeze\nSpring's gentle goodbye",
        "Mountain peak stands tall\nClouds embrace its snowy crown\nSilence speaks volumes",
        "Ocean waves retreat\nSeashells glisten on wet sand\nTide's eternal dance",
        "Autumn leaves descend\nGolden carpet marks time's path\nNature's art unfolds"
    ],
    
    'proverbs': [
        "A stitch in time saves nine.",
        "Actions speak louder than words.",
        "Better late than never arrives.",
        "Curiosity killed the cat yesterday.",
        "Every cloud has a silver lining.",
        "Fortune favors the bold heart.",
        "Good things come to those who wait.",
        "Honesty is the best policy always.",
        "Knowledge is power in life.",
        "Laughter is the best medicine."
    ]
}

def get_dataset_info(dataset_name: str) -> DatasetInfo:
    """Get information about a dataset"""
    if dataset_name not in QUANTUM_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(QUANTUM_DATASETS.keys())}")
    return QUANTUM_DATASETS[dataset_name]

def get_sample_data(dataset_name: str) -> List[str]:
    """Get sample data for a dataset"""
    if dataset_name not in SAMPLE_DATA:
        raise ValueError(f"No sample data for dataset: {dataset_name}")
    return SAMPLE_DATA[dataset_name]

def list_available_datasets() -> List[str]:
    """List all available datasets"""
    return list(QUANTUM_DATASETS.keys())