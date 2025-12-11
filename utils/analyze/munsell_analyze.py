from pathlib import Path
from typing import Optional, List
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import cosine, euclidean, cityblock
import pandas as pd
from colour.plotting import plot_chromaticity_diagram_CIE1931
from colour import xyY_to_XYZ
from colour.models import XYZ_to_CAM16LCD
from colour.difference.cam16_ucs import delta_E_CAM16UCS
from ..loaders.munsell_chains_loader import MunsellChainsLoader

class MunsellEmbeddingsAnalyzer:
    def __init__(self, embeddings_dir : Path | str = 'data/embeddings/qwen2.5_7B/describe_color/munsell_colors', csv_path : Path | str = 'data/munsell_3-3.csv'):
        self.chain_loader = MunsellChainsLoader(csv_file_path=csv_path, embeddings_directory=embeddings_dir)
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (8, 6)
        plt.rcParams['font.size'] = 14
    @staticmethod
    def _prepare_embedding_matrix(embeddings_list: list):
        """Convert list of embeddings to matrix, filtering out None values"""
        valid_embeddings = [emb for emb in embeddings_list if emb is not None]
        if not valid_embeddings:
            return None, []
        
        # Get valid indices
        valid_indices = [i for i, emb in enumerate(embeddings_list) if emb is not None]
        
        # Stack embeddings into matrix
        embedding_matrix = np.vstack(valid_embeddings)
        return embedding_matrix, valid_indices

    @staticmethod
    def _pca_by_matrix(embedding_matrix: np.ndarray):
        if embedding_matrix is None or len(embedding_matrix) == 0:
            raise RuntimeError("Empty embedding matrix provided for PCA.")
        
        embedding_matrix = np.array(embedding_matrix)
        scaler = StandardScaler()
        embedding_matrix_scaled = scaler.fit_transform(embedding_matrix)
        pca = PCA()
        pca_result = pca.fit_transform(embedding_matrix)
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
        n_components_99 = np.argmax(cumulative_variance >= 0.99) + 1
        return {
            'pca_result': pca_result,
            'explained_variance_ratio': explained_variance_ratio,
            'cumulative_variance': cumulative_variance,
            'n_components_90': n_components_90,
            'n_components_95': n_components_95,
            'n_components_99': n_components_99,
            'pca_model': pca,
            'scaler': scaler
        }

    def pca_by_embeddings(self, data: dict[str, np.ndarray]):
        """Returns PCA decomposition for embeddings in dictionary.

        Args:
            data (dict[str, np.ndarray]): the dictionary with embeddings for "lm_pooled" and "vl_pooled"
        """
        result = {}
        for embed_type in ['lm_pooled', 'vl_pooled']:
            embedding_matrix = data[embed_type]
            result[embed_type] = self._pca_by_matrix(embedding_matrix)
        return result
            
    def pca_by_chains_specification(self, variables, values, fixed_h, fixed_c, fixed_v):
        """
        Perform PCA analysis for both LM and VL embeddings and return results as dictionary.
        All arguments are lists for get_list_of_chains_by_specifications.
        
        Args:
            variables (list[str]): List indicating which parameter ('h', 'c', or 'v') is being varied in each chain.
            values (list[list[str] | None]): List of lists specifying the parameter values to vary, or None.
            fixed_h (list[str]): List of fixed hue values for each chain.
            fixed_c (list[int | None]): List of fixed chroma values for each chain.
            fixed_v (list[int | None]): List of fixed value/lightness values for each chain.

        Returns:
            dict: A dictionary containing PCA results for both language model pooled embeddings and vision-language pooled embeddings, under keys 'lm_pooled' and 'vl_pooled'.
                  Each result is either None or a dict containing:
                      - 'pca_result' (np.ndarray): Transformed data in PCA space.
                      - 'explained_variance_ratio' (np.ndarray): Variance explained by each principal component.
                      - 'cumulative_variance' (np.ndarray): Cumulative explained variance.
                      - 'n_components_90' (int): Number of components to reach 90% variance.
                      - 'n_components_95' (int): Number of components to reach 95% variance.
                      - 'n_components_99' (int): Number of components to reach 99% variance.
                      - 'pca_model' (PCA): Fitted PCA model.
                      - 'scaler' (StandardScaler): Scaler fitted to the data before PCA.
        """
        data = self.chain_loader.get_list_of_chains_by_specifications(
            variables=variables,
            values=values,
            fixed_h=fixed_h,
            fixed_c=fixed_c,
            fixed_v=fixed_v
        )
        result = {}

        for embed_type in ['lm_pooled', 'vl_pooled']:
            embedding_matrix = data[embed_type]
            if embedding_matrix is not None and len(embedding_matrix) > 0:
                result[embed_type] = self._pca_by_matrix(embedding_matrix)
            else:
                result[embed_type] = None
        return result

    @staticmethod
    def plot_pca_variance(pca_results, embedding_name):
        """Plot explained variance for PCA"""
        if embedding_name not in pca_results or pca_results[embedding_name] is None:
            return
        
        result = pca_results[embedding_name]
        components_90 = result['n_components_90']
        components_95 = result['n_components_95']
        components_99 = result['n_components_99']
        explained_var = result['explained_variance_ratio']
        cumulative_var = result['cumulative_variance']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Individual explained variance
        ax1.plot(range(1, min(51, len(explained_var) + 1)), explained_var[:50], 'b-', alpha=0.7)
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title(f'{embedding_name}: Individual Explained Variance')
        ax1.grid(True, alpha=0.3)
        
        # Cumulative explained variance
        ax2.plot(range(1, min(51, len(cumulative_var) + 1)), cumulative_var[:50], 'r-', linewidth=2)
        ax2.axhline(y=0.90, color='red', linestyle='--', label=f'90% variance, {components_90} components')
        ax2.axhline(y=0.95, color='g', linestyle='--', label=f'95% variance, {components_95} components')
        ax2.axhline(y=0.99, color='orange', linestyle='--', label=f'99% variance, {components_99} components')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance Ratio')
        ax2.set_title(f'{embedding_name}: Cumulative Explained Variance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        

    def tsne(self, variables, values, fixed_h, fixed_c, fixed_v, perplexity=7, n_iter=5000, leave_one_grey=True):
        """Perform t-SNE for both LM and VL embeddings using get_list_of_chains_by_specifications, with C=0 filtering."""

        data = self.chain_loader.get_list_of_chains_by_specifications(
            variables=variables,
            values=values,
            fixed_h=fixed_h,
            fixed_c=fixed_c,
            fixed_v=fixed_v
        )
        color_metadata = data.get('metadata', None)
        if color_metadata is None:
            raise ValueError("No metadata found for chain group.")

        results = {}
        meta = pd.DataFrame(color_metadata)
        for embedding_name in ['lm_pooled', 'vl_pooled']:
            embedding_matrix_full = data.get(embedding_name, None)
            if embedding_matrix_full is None or len(embedding_matrix_full) == 0:
                results[embedding_name] = None
                continue
            keep_indices = set(meta['csv_index'].tolist())
            if leave_one_grey:
                # Filtration
                if 'V' not in meta.columns or 'C' not in meta.columns:
                    raise ValueError(f"color_metadata must contain 'V' and 'C' columns for {embedding_name}")
                meta['csv_index'] = np.arange(len(meta))
                keep_indices = set(meta.loc[meta['C'] != 0,'csv_index'].tolist())
                for v_value, group in meta.groupby('V'):
                    c0_rows = group[group['C'] == 0]
                    if not c0_rows.empty:
                        keep_indices.add(c0_rows.iloc[0]['csv_index'])
                keep_indices = sorted(keep_indices)
            embedding_matrix_filtered = embedding_matrix_full[keep_indices]
            meta_filtered = meta.loc[meta['csv_index'].isin(keep_indices)].reset_index(drop=True)
            pca_result = self._pca_by_matrix(embedding_matrix_filtered)
            # Standardize
            # scaler = StandardScaler()
            # embedding_matrix_scaled = scaler.fit_transform(embedding_matrix_filtered)
            # t-SNE
            tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                max_iter=n_iter,
                random_state=42,
                verbose=1,
                min_grad_norm=1e-8,
                n_iter_without_progress=1000
            )
            tsne_result = tsne.fit_transform(pca_result['pca_result'])
            results[embedding_name] = {
                'tsne_result': tsne_result,
                'tsne_model': tsne,
                'scaler': pca_result['scaler'],
                'pca_model': pca_result['pca_model'],
                'meta_filtered': meta_filtered
            }
        return results
    
    @staticmethod
    def plot_tsne_results(tsne_results, embedding_name):
        """Plot t-SNE results using filtered metadata"""
        if embedding_name not in tsne_results or tsne_results[embedding_name] is None:
            return
        
        tsne_result = tsne_results[embedding_name]['tsne_result']
        meta = tsne_results[embedding_name]['meta_filtered']

        # Извлекаем данные для цветовой визуализации
        hues = meta.get('H', pd.Series(['Unknown'] * len(meta)))
        chromas = meta.get('C', pd.Series([0] * len(meta)))
        values = meta.get('V', pd.Series([0] * len(meta)))
        rgb_vals = meta.get('RGB', pd.Series([0] * len(meta)))


        # Фигура
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f't-SNE Visualization for {embedding_name}', fontsize=16)

        # --- 1. По Hue ---
        unique_hues = list(set(hues))
        hue_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_hues)))
        hue_color_map = {hue: hue_colors[i] for i, hue in enumerate(unique_hues)}

        for hue in unique_hues:
            mask = hues == hue
            axes[0,0].scatter(tsne_result[mask, 0], tsne_result[mask, 1],
                            c=[hue_color_map[hue]], label=hue, alpha=1, s=100)
        axes[0,0].set_title('Colored by Hue')
        axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0,0].grid(True, alpha=0.3)

        # --- 2. По Chroma ---
        sc1 = axes[0,1].scatter(tsne_result[:, 0], tsne_result[:, 1],
                                c=chromas, cmap='viridis', alpha=0.7, s=100)
        axes[0,1].set_title('Colored by Chroma')
        plt.colorbar(sc1, ax=axes[0,1])
        axes[0,1].grid(True, alpha=0.3)
        # --- 3. По Value ---
        sc2 = axes[1,0].scatter(tsne_result[:, 0], tsne_result[:, 1],
                                c=values, cmap='plasma', alpha=0.7, s=100)
        axes[1,0].set_title('Colored by Value')
        plt.colorbar(sc2, ax=axes[1,0])
        axes[1,0].grid(True, alpha=0.3)

        # --- 4. С подписями Munsell Notation ---
        
        
        for h, c, v, rgb, i in zip(hues, chromas, values, rgb_vals, range(len(hues))):
            axes[1,1].scatter(tsne_result[i, 0], tsne_result[i, 1], c=rgb, s=200)
            notation = f"{h}/{v} {c}"
            axes[1,1].annotate(str(notation), (tsne_result[i, 0], tsne_result[i, 1]),
                            xytext=(5, 5), textcoords='offset points', fontsize=10)
        axes[1,1].set_title('With Munsell Notations')
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
    
    def plot_chromaticity_diagram(self, variables, values, fixed_h, fixed_c, fixed_v, show_labels=False, show=False):
        """
        Plot xy chromaticity for all points (not per-chain) using colour-science CIE1931 diagram.
        Args:
            variables, values, fixed_h, fixed_c, fixed_v: lists for get_list_of_chains_by_specifications
            show_labels: if True, plot V/C/H info for each point.
        """
        data = self.chain_loader.get_list_of_chains_by_specifications(
            variables=variables,
            values=values,
            fixed_h=fixed_h,
            fixed_c=fixed_c,
            fixed_v=fixed_v,
        )
        metadata = data['metadata']
        print(metadata)
        if not metadata:
            print("No points found for plotting.")
            return
        x = [entry['xyY'][0] for entry in metadata]
        y = [entry['xyY'][1] for entry in metadata]
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_chromaticity_diagram_CIE1931(standalone=False, axes=ax)
        ax.scatter(x, y, s=70, color='red', edgecolor='k', alpha=0.85, label='Points')
        if show_labels:
            for entry in metadata:
                label = f"H={entry['H']}\nC={entry['V']}\nV={entry['C']}"
                ax.text(entry['xyY'][0], entry['xyY'][1], label, fontsize=8, alpha=0.7)
        ax.set_title('All points on CIE 1931 Chromaticity Diagram')
        ax.legend()
        plt.tight_layout()
        if show:
            plt.show()
            
        return fig, ax
    
    def calculate_distances_matrix(self,
                                   variable: str,
                                   values: Optional[list[int | str]],
                                   fixed_h: Optional[str],
                                   fixed_c: Optional[int],
                                   fixed_v: Optional[int],
                                   return_rgb: bool=False,
                                   cumulative: bool=False):
        """Calculates the distance matrix by different variables: cosine distance of VL and LM embeddings, sRGB euclidean distance, munsell Manhattan distance between variables in a chain, CIE CAM 16USC.

        Args:
            variable (str): 'h', 'c', 'v
            values (list[int]): List of variables in chain.
            fixed_h (str): Fixed H munsell value ('2.5R').
            fixed_c (int): Fixed C munsell value.
            fixed_v (int): Fixed V munsell value.
            return_rgb (bool): True or False
            cumulative (bool): True or False. To calculate cumulative distance in chain or absolute.

        Returns:
            dict: Dict with keys "srgb","vl_cosine","lm_cosine","cam","munsell" and according matrix with distances: in a i row distance from i-th to all. If return_srgb is true returns tuple (dict, srgb_array)
        """
        chain = self.chain_loader.get_chain_by_specification(variable, values, fixed_h, fixed_c, fixed_v)
        metadata = pd.DataFrame(chain['metadata'])
        vl_embeds = np.array(chain['vl_pooled'])
        lm_embeds = np.array(chain['lm_pooled'])
        srgb_colors_arr = metadata['RGB'].to_numpy()
        variable_arr = metadata[variable.upper()].to_numpy()
        xyY_arr = metadata['xyY'].to_numpy()

        distances_matrix = {
            "srgb": np.zeros((len(metadata), len(metadata))),
            "vl_cosine": np.zeros((len(metadata), len(metadata))),
            "lm_cosine": np.zeros((len(metadata), len(metadata))),
            "cam": np.zeros((len(metadata), len(metadata))),
            "munsell": np.zeros((len(metadata), len(metadata))),
            "chain_info" : {"variable" : [variable, values]}
            }
        
        if cumulative:
            cumulative_totals = {
                "srgb": 0.0,
                "vl_cosine": 0.0,
                "lm_cosine": 0.0,
                "cam": 0.0,
                "munsell": 0.0
            }
            for idx in range(1, len(metadata)):
                prev_idx = idx - 1

                prev_srgb = srgb_colors_arr[prev_idx]
                prev_xyy = xyY_arr[prev_idx]
                prev_vl_emb = vl_embeds[prev_idx]
                prev_lm_emb = lm_embeds[prev_idx]
                prev_cam = XYZ_to_CAM16LCD(xyY_to_XYZ(prev_xyy))
                prev_munsell_variable = variable_arr[prev_idx]

                current_srgb = srgb_colors_arr[idx]
                current_xyy = xyY_arr[idx]
                current_vl_emb = vl_embeds[idx]
                current_lm_emb = lm_embeds[idx]
                current_cam = XYZ_to_CAM16LCD(xyY_to_XYZ(current_xyy))
                current_munsell_variable = variable_arr[idx]

                srgb_distance = euclidean(prev_srgb, current_srgb)
                vl_cosine_distance = cosine(prev_vl_emb, current_vl_emb)
                lm_cosine_distance = cosine(prev_lm_emb, current_lm_emb)
                cam_distance = delta_E_CAM16UCS(prev_cam, current_cam)
                munsell_distance = np.abs(prev_munsell_variable - current_munsell_variable)

                cumulative_totals["srgb"] += srgb_distance
                cumulative_totals["vl_cosine"] += vl_cosine_distance
                cumulative_totals["lm_cosine"] += lm_cosine_distance
                cumulative_totals["cam"] += cam_distance
                cumulative_totals["munsell"] += munsell_distance

                distances_matrix["srgb"][0, idx] = cumulative_totals["srgb"]
                distances_matrix["vl_cosine"][0, idx] = cumulative_totals["vl_cosine"]
                distances_matrix["lm_cosine"][0, idx] = cumulative_totals["lm_cosine"]
                distances_matrix["cam"][0, idx] = cumulative_totals["cam"]
                distances_matrix["munsell"][0, idx] = cumulative_totals["munsell"]
        else:
            for first_idx in range(len(metadata)):
                for second_idx in range(len(metadata)):
                    first_srgb = srgb_colors_arr[first_idx]
                    first_xyy = xyY_arr[first_idx]
                    first_vl_emb = vl_embeds[first_idx]
                    first_lm_emb = lm_embeds[first_idx]
                    first_cam = XYZ_to_CAM16LCD(xyY_to_XYZ(first_xyy))
                    first_munsell_variable = variable_arr[first_idx]

                    second_srgb = srgb_colors_arr[second_idx]
                    second_xyy = xyY_arr[second_idx]
                    second_vl_emb = vl_embeds[second_idx]
                    second_lm_emb = lm_embeds[second_idx]
                    second_cam = XYZ_to_CAM16LCD(xyY_to_XYZ(second_xyy))
                    second_munsell_variable = variable_arr[second_idx]

                    distances_matrix['srgb'][first_idx, second_idx] = euclidean(first_srgb, second_srgb)
                    distances_matrix['vl_cosine'][first_idx, second_idx] = cosine(first_vl_emb, second_vl_emb)
                    distances_matrix['lm_cosine'][first_idx, second_idx] = cosine(first_lm_emb, second_lm_emb)
                    distances_matrix['cam'][first_idx, second_idx] = delta_E_CAM16UCS(first_cam, second_cam)
                    distances_matrix['munsell'][first_idx, second_idx] = np.abs(first_munsell_variable - second_munsell_variable)
        if return_rgb:
            return distances_matrix, srgb_colors_arr
        return distances_matrix

    def calculate_list_distances_matrix(self,
                                        variables: list[str],
                                        values: list[Optional[list[str | int]]],
                                        fixed_h: list[Optional[str]],
                                        fixed_c: list[Optional[int]],
                                        fixed_v: list[Optional[int]],
                                        return_rgb: bool=False,
                                        cumulative: bool=False):
        self.chain_loader.get_list_of_chains_by_specifications(variables, values, fixed_h, fixed_c, fixed_v)
        list_of_distances_matrix = []
        rgb_list = []
        rgb = None
        
        for var, val, h_fix, c_fix, v_fix in zip(variables, values, fixed_h, fixed_c, fixed_v):
            if return_rgb:
                matrix, rgb = self.calculate_distances_matrix(var, val, h_fix, c_fix, v_fix, return_rgb, cumulative)
            else:
                matrix = self.calculate_distances_matrix(var, val, h_fix, c_fix, v_fix, return_rgb, cumulative)
            list_of_distances_matrix.append(matrix)
            rgb_list.append(rgb)
            
        if return_rgb:
            return list_of_distances_matrix, rgb_list
        return list_of_distances_matrix
        

    @staticmethod
    def plot_distance_matrix_result(distances_dict, init_idx=0, x : str="munsell", rgb_arr=None, show=True, fig = None, axes=None):
        if rgb_arr is None:
            rgb_arr = []
        if axes is not None:
            ax1, ax2 = axes
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.plot(distances_dict[x][init_idx, 1:], distances_dict['vl_cosine'][init_idx, 1:], linewidth=2, linestyle='--')
        ax1.scatter(distances_dict[x][init_idx, 1:], distances_dict['vl_cosine'][init_idx, 1:], c=rgb_arr[1:], s=300, alpha=1)
        ax1.set_xlabel(x)
        ax1.set_ylabel('VL cosine distances')
        ax1.grid(True)
        
        ax2.scatter(distances_dict[x][init_idx, 1:], distances_dict['lm_cosine'][init_idx, 1:], c=rgb_arr[1:], s=300, alpha=1)
        ax2.plot(distances_dict[x][init_idx, 1:], distances_dict['lm_cosine'][init_idx, 1:], linewidth=2, linestyle='--')
        ax2.set_xlabel(x)
        ax2.set_ylabel('LM cosine distances')
        ax2.grid(True)
        
        if show:
            plt.show()
        return fig, axes
    
    def plot_list_distances_maxtrix_result(self, list_distances_dict, init_idx=0, x : str="munsell", rgb_arr=None, show=True):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        for idx, dist_dict in enumerate(list_distances_dict):
            self.plot_distance_matrix_result(dist_dict, init_idx, x, rgb_arr[idx] if rgb_arr else None, False, fig, (ax1, ax2))
        if show:
            plt.show()        
        return fig, (ax1, ax2)
    
    def draw_munsell_chain(self, variable, values, fixed_h, fixed_c, fixed_v):
        chain = self.chain_loader.get_chain_by_specification(variable, values, fixed_h, fixed_c, fixed_v)['metadata']
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(8,6))
        
        # Calculate swatch dimensions
        n_colors = len(chain)
        swatch_width = 2
        swatch_height = 10
        
        # Draw each color swatch
        for i, color in enumerate(chain):
            # Convert xyY to RGB using normalized Y
            
            rgb = color['RGB']
            # Create rectangle for color swatch
            rect = patches.Rectangle((i, 0), swatch_width, swatch_height, 
                                facecolor=rgb, edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
            
            # Add label
            if variable == 'h':
                label = f"{color['H']}"
            elif variable == 'v':
                label = f"V{color['V']}\n{color['H']}/C{color['C']}"
            elif variable == 'c':
                label = f"C{color['C']}\n{color['H']}/V{color['V']}"
            
            ax.text(i + swatch_width/2, -0.1, label, ha='center', va='top', 
                    fontsize=12, rotation=45)
        
        # Set axis properties
        ax.set_xlim(0, n_colors)
        ax.set_ylim(-0.3, 5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add title
        title_parts = []
        if fixed_h: title_parts.append(f"Hue: {fixed_h}")
        if fixed_v: title_parts.append(f"Value: {fixed_v}")
        if fixed_c: title_parts.append(f"Chroma: {fixed_c}")
        title_parts.append(f"Varying: {variable}")
        
        ax.set_title(" / ".join(title_parts), fontsize=12, pad=20)
        
        plt.tight_layout()
        return fig, ax
