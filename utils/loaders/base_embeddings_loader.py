"""
Base Embeddings Loader
---------------------
General-purpose embeddings loader that parses color information from manifest files.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Union
import numpy as np


class EmbeddingsLoader:
    """
    Base class for loading embeddings and parsing color information from manifests.
    
    Example of usage: 
    
    from utils.loaders.base_embeddings_loader import EmbeddingsLoader
    loader = EmbeddingsLoader('data/embeddings/qwen2.5_7B/human_like/munsell_colors')
    embeddings_indexes = ['1', '2', '3']
    
    embeds_asn_info = loader.get_embeddings_by_indices(embeddings_indexes)
    
    # OR
    
    embeddings = loader.load(embeddings_indexes)
    color_info = loader.load(embedding_indexes)
    """
    
    def __init__(self, embeddings_directory: Union[str, Path]):
        """
        Initialize the embeddings loader.
        
        Args:
            embeddings_directory: Path to directory containing embedding subdirectories
        """
        self.embeddings_dir = Path(embeddings_directory)
        
        if not self.embeddings_dir.exists():
            raise FileNotFoundError(f"Embeddings directory not found: {self.embeddings_dir}")
        
        # Load embeddings manifests and create index mapping
        self.embeddings_index = self._load_embeddings_index()
    
    def _load_embeddings_index(self) -> Dict[str, Dict[str, Any]]:
        """Load embeddings manifests and create index mapping."""
        embeddings_index = {}
        
        # Find all subdirectories with numeric names
        subdirs = sorted(
            [d for d in self.embeddings_dir.iterdir() 
             if d.is_dir() and d.name.isdigit()],
            key=lambda d: int(d.name)
        )
        
        for subdir in subdirs:
            manifest_path = subdir / "manifest.json"
            if manifest_path.exists():
                with manifest_path.open("r", encoding="utf-8") as f:
                    manifest = json.load(f)
                
                # Extract embedding paths and metadata
                embeddings_index[subdir.name] = {
                    'manifest': manifest,
                    'lm_pooled_path': subdir / "lm_pooled_mean.npy",
                    'vision_pooled_path': subdir / "vision_pooled_mean.npy",
                    'visual_token_lens_path': subdir / "visual_token_lens.npy",
                    'csv_row': manifest.get('csv_row', {}),
                    'xyY': manifest.get('xyY', {}),
                    'RGB': manifest.get('RGB', {}),
                    'image_path': manifest.get('image', ''),
                    'answer': manifest.get('answer', ''),
                    'prompt': manifest.get('prompt', '')
                }
        
        return embeddings_index
    
    def get_colors_info(self, embedding_indexes: List[str]) -> Dict[str, Any]:
        """
        Get comprehensive color information for a specific embedding index.
        
        Args:
            embedding_index: Index of the embedding
            
        Returns:
            Dictionary with color information including xyY, RGB, HSV
        """
        result = {}
        
        for embedding_index in embedding_indexes:
            if embedding_index not in self.embeddings_index:
                raise KeyError(f"Embedding index {embedding_index} not found")
            
            emb_data = self.embeddings_index[embedding_index]
            
            # Extract color information
            color_info = {
                'embedding_index': embedding_index,
                'xyY': emb_data['xyY'],
                'RGB': emb_data['RGB'],
                'image_path': emb_data['image_path'],
                'answer': emb_data['answer'],
                'prompt': emb_data['prompt']
            }
            
            result[embedding_index] = color_info
        
        return result
    
    def load_embeddings(self, embedding_indices: List[str] | None = None) -> Dict[str, np.ndarray]:
        """
        Load embeddings for given indices.
        
        Args:
            embedding_indices: List of embedding indices to load. If None returns all embeddings.
            
        Returns:
            Dictionary with 'lm_pooled' and 'vl_pooled' arrays
        """
        lm_embeddings = []
        vl_embeddings = []
        
        if embedding_indices is None:
            embedding_indices = list(self.embeddings_index.keys())
        
        for idx in embedding_indices:
            if idx in self.embeddings_index:
                emb_data = self.embeddings_index[idx]
                
                # Load LM pooled embeddings
                if emb_data['lm_pooled_path'].exists():
                    lm_emb = np.load(emb_data['lm_pooled_path'])
                    lm_embeddings.append(lm_emb.flatten())
                
                # Load Vision pooled embeddings
                if emb_data['vision_pooled_path'].exists():
                    vl_emb = np.load(emb_data['vision_pooled_path'])
                    vl_embeddings.append(vl_emb.flatten())
        
        result = {}
        if lm_embeddings:
            result['lm_pooled'] = np.stack(lm_embeddings, axis=0)
        if vl_embeddings:
            result['vl_pooled'] = np.stack(vl_embeddings, axis=0)
        
        return result
    
    def get_embeddings_by_indices(self, indices: List[str]) -> Dict[str, Any]:
        """
        Get embeddings and color information for specific indices.
        
        Args:
            indices: List of embedding indices
            
        Returns:
            Dictionary with embeddings and metadata
        """
        embeddings = self.load_embeddings(indices)
        metadata = self.get_colors_info(indices)

        return {
            'lm_pooled': embeddings.get('lm_pooled'),
            'vl_pooled': embeddings.get('vl_pooled'),
            'metadata': metadata
        }
    
    def __len__(self) -> int:
        """Return the number of available embeddings."""
        return len(self.embeddings_index)
    
    def __iter__(self):
        """Iterate over embedding indices."""
        return iter(self.embeddings_index.keys())
    
    def __contains__(self, embedding_index: str) -> bool:
        """Check if embedding index exists."""
        return embedding_index in self.embeddings_index
