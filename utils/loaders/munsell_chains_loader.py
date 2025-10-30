"""
Munsell Chains Loader
--------------------
Specialized loader for Munsell color chains that inherits from EmbeddingsLoader.
Adds CSV parsing, MunsellColor integration, and chain generation functionality.
"""
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union
import pandas as pd
import numpy as np
from .base_embeddings_loader import EmbeddingsLoader
from ..color.gen_color import MunsellColor


class MunsellChainsLoader(EmbeddingsLoader):
    """
    Specialized loader for Munsell color chains that inherits from EmbeddingsLoader.
    Adds CSV parsing, MunsellColor integration, and chain generation functionality.
    
    Example of usage: 
    
    from utils.loaders.munsell_chains_loader import MunsellChainsLoader
    loader = MunsellChainsLoader('data/munsell_3-3.csv', 'data/embeddings/qwen2.5_7B/human_like/munsell_colors')
    embeddings_and_metadata_dict_1 = loader.get_chain_by_specification(variable='v', fixed_h='2.5R', fixed_c='4') # for all possible varying values
    embeddings_and_metadata_dict_2 = loader.get_chain_by_specification(variable='v', values=[1, 2, 3] fixed_h='2.5R', fixed_c='4') # for specifies  varying values
    
    # see variants of H, V, C
    statistics = loader.def get_color_statistics()
    """
    
    def __init__(self, csv_file_path: Union[str, Path], embeddings_directory: Union[str, Path]):
        """
        Initialize the Munsell chains loader.
        
        Args:
            csv_file_path: Path to the CSV file with color data (H,V,C,x,y,Y columns)
            embeddings_directory: Path to directory containing embedding subdirectories
        """
        # Initialize base class
        super().__init__(embeddings_directory)
        
        # Load CSV data
        self.csv_path = Path(csv_file_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        self.color_table = self._load_csv_data()
    
    def _load_csv_data(self) -> pd.DataFrame:
        """Load and process CSV data with color information."""
        df = pd.read_csv(self.csv_path)
        
        # Ensure required columns exist
        required_cols = ['H', 'V', 'C', 'x', 'y', 'Y']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Add RGB conversion using the existing function
        rgb_values = []
        for _, row in df.iterrows():
            rgb = self._xyY_to_rgb(row['x'], row['y'], row['Y'])
            rgb_values.append(rgb)
        
        df['R'] = [rgb[0] for rgb in rgb_values]
        df['G'] = [rgb[1] for rgb in rgb_values]
        df['B'] = [rgb[2] for rgb in rgb_values]
        
        # Add index column for mapping to embeddings
        df['index'] = df.index
        
        return df
    
    def _xyY_to_rgb(self, x: float, y: float, Y: float) -> Tuple[float, float, float]:
        """Convert xyY to RGB using the existing conversion function."""
        import colour
        
        xyY = np.array([x, y, Y], dtype=float)
        # Normalize Y if it's in 0-100 range
        xyY[2] = xyY[2] / 102.5
        
        XYZ = colour.xyY_to_XYZ(xyY)
        srgb = colour.XYZ_to_sRGB(XYZ, apply_cctf_encoding=True)
        return tuple(srgb)
    
    
    def _normalize_hue(self, hue_val) -> str:
        """Normalize hue value to match CSV format."""
        if isinstance(hue_val, str):
            # Convert formats like "5R" to "5.0R" to match CSV
            if hue_val == 'N':
                return 'N'
            # Handle formats like "5R", "2.5Y", etc.
            import re
            match = re.match(r'^(\d+(?:\.\d+)?)([A-Z]+)$', hue_val)
            if match:
                number, letter = match.groups()
                return f"{number}.0{letter}" if '.' not in number else hue_val
        elif isinstance(hue_val, (int, float)):
            # Convert numeric hue to string format
            return f"{hue_val}.0R"  # Default to R if no letter specified
        return str(hue_val)
    
    def find_colors_in_csv(self, munsell_colors: List[MunsellColor]) -> List[Dict[str, Any]]:
        """
        Find colors in CSV table that match the given MunsellColor objects.
        
        Args:
            munsell_colors: List of MunsellColor objects
            
        Returns:
            List of dictionaries with CSV row data and embedding index
        """
        found_colors = []
        
        for munsell_color in munsell_colors:
            # Convert MunsellColor to searchable format
            h_val = self._normalize_hue(munsell_color.h)
            c_val = int(munsell_color.c)
            v_val = int(munsell_color.v)
            
            # Find matching rows in CSV
            matches = self.color_table[
                (self.color_table['H'] == h_val) &
                (self.color_table['C'] == c_val) &
                (self.color_table['V'] == v_val)
            ]
            
            if not matches.empty:
                for _, row in matches.iterrows():
                    color_data = {
                        'munsell_color': munsell_color,
                        'csv_index': int(row['index']),
                        'xyY': (row['x'], row['y'], row['Y']),
                        'RGB': (row['R'], row['G'], row['B']),
                        'H': row['H'],
                        'C': int(row['C']),
                        'V': int(row['V']),
                        'embedding_index': None  # Will be set if embedding exists
                    }
                    
                    # Check if embedding exists for this index
                    csv_idx_str = str(int(row['index']))
                    if csv_idx_str in self.embeddings_index:
                        color_data['embedding_index'] = csv_idx_str
                    
                    found_colors.append(color_data)
        
        return found_colors
    
    def get_chain(self, munsell_colors: List[MunsellColor]) -> Dict[str, Any]:
        """
        Get chain data for a list of MunsellColor objects.
        
        Args:
            munsell_colors: List of MunsellColor objects
            
        Returns:
            Dictionary with embeddings and metadata
        """
        # Find colors in CSV
        found_colors = self.find_colors_in_csv(munsell_colors)
        
        if not found_colors:
            return {'lm_pooled': None, 'vl_pooled': None, 'metadata': []}
        
        # Extract embedding indices
        embedding_indices = [color['embedding_index'] for color in found_colors 
                           if color['embedding_index'] is not None]
        
        # Load embeddings using parent class method
        embeddings = self.load_embeddings(embedding_indices)
        
        # Prepare metadata
        metadata = []
        for color in found_colors:
            metadata_entry = {
                'munsell_color': color['munsell_color'],
                'xyY': color['xyY'],
                'RGB': color['RGB'],
                'H': color['H'],
                'C': color['C'],
                'V': color['V'],
                'csv_index': color['csv_index'],
                'has_embedding': color['embedding_index'] is not None
            }
            metadata.append(metadata_entry)
        
        return {
            'lm_pooled': embeddings.get('lm_pooled'),
            'vl_pooled': embeddings.get('vl_pooled'),
            'metadata': metadata
        }
    
    def get_chain_by_specification(self, variable: str, values: (List | None) = None,
                                 fixed_h=None, fixed_c: int | None = None, fixed_v: int | None = None) -> Dict[str, Any]:
        """
        Generate a chain using MunsellColor.chain and get embeddings.
        
        Args:
            variable: Which parameter to vary ('h', 'c', or 'v')
            values: Values for the varying parameter
            fixed_h: Fixed H value when variable != 'h'
            fixed_c: Fixed C value when variable != 'c'
            fixed_v: Fixed V value when variable != 'v'
            
        Returns:
            Dictionary with embeddings and metadata
        """
        if values is None:
            values = np.arange(0, 40)
        # Generate MunsellColor chain
        munsell_colors = MunsellColor.chain(variable, values, fixed_h, fixed_c, fixed_v)
        
        # Get chain data
        return self.get_chain(munsell_colors)
    
    def get_list_of_chains_by_specifications(self, variables: list[str], values: List[List[str] | None], fixed_h: List[str], fixed_c: List[int | None], fixed_v: List[int | None]):
        """Returns multiple chains information.

        Args:
            variables (list[str]): List of which parameter to vary 'h', 'c' or 'v'.
            values (List[List[str]  |  None]): List of list of variables parameters or None.
            fixed_h (List[str]): List of fixed H values.
            fixed_c (List[int  |  None]): List of fixed C values.
            fixed_v (List[int  |  None]): List of fixed V values.
        """
        all_metadata = []
        lm_pooled_list = []
        vl_pooled_list = []
        for variable, vals, h_fix, c_fix, v_fix in zip(variables, values, fixed_h, fixed_c, fixed_v):
            # Ensure only two fixed values, set the one being varied to None
            if variable == 'h':
                chain = self.get_chain_by_specification(
                    variable=variable,
                    values=vals,
                    fixed_h=None,
                    fixed_c=c_fix,
                    fixed_v=v_fix
                )
            elif variable == 'c':
                chain = self.get_chain_by_specification(
                    variable=variable,
                    values=vals,
                    fixed_h=h_fix,
                    fixed_c=None,
                    fixed_v=v_fix
                )
            elif variable == 'v':
                chain = self.get_chain_by_specification(
                    variable=variable,
                    values=vals,
                    fixed_h=h_fix,
                    fixed_c=c_fix,
                    fixed_v=None
                )
            else:
                raise ValueError(f"Invalid variable: {variable}")
            if chain['metadata']:
                all_metadata.extend(chain['metadata'])
            if chain['lm_pooled'] is not None:
                lm_pooled_list.append(chain['lm_pooled'])
            if chain['vl_pooled'] is not None:
                vl_pooled_list.append(chain['vl_pooled'])
        lm_pooled = np.vstack(lm_pooled_list) if lm_pooled_list else None
        vl_pooled = np.vstack(vl_pooled_list) if vl_pooled_list else None
        return {
            'metadata': all_metadata,
            'lm_pooled': lm_pooled,
            'vl_pooled': vl_pooled
        }
    
    def get_all_available_colors(self) -> List[Dict[str, Any]]:
        """Get all colors that have both CSV data and embeddings."""
        available_colors = []
        
        for _, row in self.color_table.iterrows():
            idx_str = str(int(row['index']))
            if idx_str in self.embeddings_index:
                color_data = {
                    'csv_index': int(row['index']),
                    'H': row['H'],
                    'C': int(row['C']),
                    'V': int(row['V']),
                    'xyY': (row['x'], row['y'], row['Y']),
                    'RGB': (row['R'], row['G'], row['B']),
                    'munsell_spec': f"{row['H']} {row['V']}/{row['C']}",
                    'embedding_index': idx_str
                }
                available_colors.append(color_data)
        
        return available_colors
    
    def get_all_available_embeddings(self) -> Dict[str, Any]:
        return self.get_embeddings_by_indices(list(self.embeddings_index.keys()))
    
    def get_color_statistics(self) -> Dict[str, Any]:
        """Get statistics about available colors."""
        available_colors = self.get_all_available_colors()
        
        if not available_colors:
            return {}
        
        return {
            'total_colors': len(available_colors),
            'unique_hues': set(color['H'] for color in available_colors),
            'unique_chromas': set(color['C'] for color in available_colors),
            'unique_values': set(color['V'] for color in available_colors)
        }