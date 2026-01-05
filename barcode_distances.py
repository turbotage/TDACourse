def plot_distance_heatmap(D, names, types, title, fname):
	import matplotlib.pyplot as plt
	import numpy as np
	plt.figure(figsize=(12, 10))
	im = plt.imshow(D, aspect='auto', cmap='viridis')
	plt.colorbar(im, fraction=0.046, pad=0.04)
	# Show every label, but rotate for readability
	tick_labels = [f"{n}\n{t}" for n, t in zip(names, types)]
	plt.xticks(np.arange(len(names)), tick_labels, rotation=90, fontsize=7)
	plt.yticks(np.arange(len(names)), tick_labels, fontsize=7)
	plt.title(title)
	plt.tight_layout()
	plt.savefig(fname)
	print(f"Saved distance heatmap to {fname}")
	plt.close()
# Top-level function for multiprocessing distance computation
def pd_pair_distance(args):
	i, j, diagrams, metric, homology_dim = args
	from persim import wasserstein, bottleneck
	def get_dist(d1, d2, metric):
		if metric == "wasserstein":
			return wasserstein(d1, d2)
		elif metric == "bottleneck":
			return bottleneck(d1, d2)
		else:
			raise ValueError("Unknown metric: " + metric)
	d1 = diagrams[i][homology_dim]
	d2 = diagrams[j][homology_dim]
	return get_dist(d1, d2, metric)
# Script to cluster persistence diagrams using Wasserstein and Bottleneck distances

import os
import pickle
import numpy as np
from distances import models, ModelData
from functools import partial
import multiprocessing

# Try to import persim, if not available, raise informative error
try:
    from persim import wasserstein, bottleneck
except ImportError:
    raise ImportError("persim is required. Install with 'pip install persim'.")

from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS, TSNE
try:
    import umap
except ImportError:
    umap = None
    print("Warning: umap-learn not installed. Install with 'pip install umap-learn'")
import matplotlib.pyplot as plt
from tqdm import tqdm

def compute_and_cache_diagram(data, model_name, batch_idx, typ, start_idx, cache_dir):
	"""Compute or load a persistence diagram for a given model, batch, type, and start index."""
	cache_file = os.path.join(cache_dir, f"{model_name}_batch{batch_idx}_{typ}_start{start_idx}.npz")
	if os.path.exists(cache_file + '.pkl'):
		try:
			with open(cache_file + '.pkl', 'rb') as f:
				dgms = pickle.load(f)
			return dgms
		except Exception as e:
			print(f"Error loading cache {cache_file}.pkl: {e}")
	try:
		from ripser import ripser
		window = data[start_idx:start_idx+500, ...]
		arr = window.numpy()
		if arr.ndim != 2:
			print(f"Skipping {model_name} batch {batch_idx} {typ} start {start_idx}: shape {arr.shape}, type {type(arr)} (expected 2D)")
			return None
		if arr.dtype == object:
			print(f"DEBUG: {model_name} batch {batch_idx} {typ} start {start_idx} arr.dtype={arr.dtype}, arr[:5]={arr[:5]}")
			print(f"Skipping {model_name} batch {batch_idx} {typ} start {start_idx}: dtype object (expected float32/float64)")
			return None
		if np.isnan(arr).any() or np.isinf(arr).any():
			print(f"Skipping {model_name} batch {batch_idx} {typ} start {start_idx}: contains NaN or inf")
			return None
		dgms = ripser(arr, maxdim=2)["dgms"]
		with open(cache_file + '.pkl', 'wb') as f:
			pickle.dump(dgms, f)
		return dgms
	except Exception as e:
		print(f"Error computing diagram for {model_name} batch {batch_idx} {typ} start {start_idx}: {e}")
		return None

def load_all_diagrams(model_dir="model_data", cache_dir="diagrams_cache", start_indices=[0, 500], max_batches=None, n_jobs=None):
	"""Load all PCA and VAE persistence diagrams from model_data/model_*.pkl, with caching and parallelism."""
	all_diagrams = []
	model_names = []
	types = []
	os.makedirs(cache_dir, exist_ok=True)

# New: Load all diagrams from cache for clustering
def load_all_diagrams_from_cache(cache_dir="diagrams_cache"):
	"""Load all persistence diagrams from diagrams_cache/ for distance computation and clustering."""
	all_diagrams = []
	model_names = []
	types = []
	import glob
	cache_files = sorted(glob.glob(os.path.join(cache_dir, "*.pkl")))
	for cache_file in tqdm(cache_files, desc="Loading cached diagrams"):
		try:
			with open(cache_file, 'rb') as f:
				dgms = pickle.load(f)
			all_diagrams.append(dgms)
			# Use filename (without extension) as label
			fname = os.path.basename(cache_file)[:-4]
			model_names.append(fname)
			# Extract type (PCA/VAE) from filename
			if "_PCA_" in fname:
				types.append("PCA")
			elif "_VAE_" in fname:
				types.append("VAE")
			else:
				types.append("Unknown")
		except Exception as e:
			print(f"Error loading cache {cache_file}: {e}")
	return all_diagrams, model_names, types
	tasks = []
	# Multiprocessing for diagram computation
	n_jobs = multiprocessing.cpu_count() if n_jobs is None else n_jobs
	if tasks:
		with multiprocessing.Pool(n_jobs) as pool:
			list(tqdm(pool.starmap(compute_and_cache_diagram, tasks), total=len(tasks), desc="Computing diagrams"))
	# Now load all diagrams (cached or just computed)
	all_diagrams = []
	model_names = []
	types = []
	for model_name in models:
		pkl_path = os.path.join(model_dir, f"model_{model_name}.pkl")
		if not os.path.exists(pkl_path):
			continue
		with open(pkl_path, "rb") as f:
			model_data = pickle.load(f)
		n_batches = model_data.img_pca.shape[0] if max_batches is None else min(model_data.img_pca.shape[0], max_batches)
		for j in range(n_batches):
			for start_idx in start_indices:
				for typ in ["PCA", "VAE"]:
					cache_file = os.path.join(cache_dir, f"{model_name}_batch{j}_{typ}_start{start_idx}.pkl")
					if os.path.exists(cache_file):
						try:
							with open(cache_file, 'rb') as f:
								dgms = pickle.load(f)
							all_diagrams.append(dgms)
							model_names.append(f"{model_name}_batch{j}_{typ}_start{start_idx}")
							types.append(f"{typ}_start{start_idx}")
						except Exception as e:
							print(f"Error loading cache {cache_file}: {e}")
	return all_diagrams, model_names, types

def compute_distance_matrix(diagrams, metric="wasserstein", homology_dim=1):
	"""Compute pairwise distance matrix for diagrams using given metric and homology dimension.
	homology_dim can be 0, 1, 2, or 'all' to sum over all available dimensions."""
	n = len(diagrams)
	D = np.zeros((n, n))
	cache_file = f"distance_matrix_{metric}_h{homology_dim}.npy"
	if os.path.exists(cache_file):
		print(f"Loading cached distance matrix from {cache_file}")
		D = np.load(cache_file)
		return D
	pairs = [(i, j, diagrams, metric, homology_dim) for i in range(n) for j in range(i+1, n)]
	with multiprocessing.Pool() as pool:
		dists = list(tqdm(pool.map(pd_pair_distance, pairs), total=len(pairs), desc=f"Computing {metric} distances (h={homology_dim})"))
	idx = 0
	for i in tqdm(range(n), desc="Filling distance matrix"):
		for j in range(i+1, n):
			D[i, j] = D[j, i] = dists[idx]
			idx += 1
	np.save(cache_file, D)
	print(f"Saved distance matrix to {cache_file}")
	return D
# Other clustering/visualization methods you can try (using the cached distance matrix):
# - DBSCAN (sklearn.cluster.DBSCAN, with metric='precomputed')
# - SpectralClustering (sklearn.cluster.SpectralClustering, with affinity='precomputed')
# - AgglomerativeClustering (with different linkage)
# - t-SNE or UMAP for 2D embedding (on D)


def cluster_and_report(D, model_names, n_clusters=2, method="ward"):
	"""Cluster using AgglomerativeClustering and print cluster assignments."""
	# Ward linkage does not support precomputed distances; use 'average' if precomputed
	linkage = method
	if method == "ward":
		linkage = "average"
	clustering = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage=linkage)
	labels = clustering.fit_predict(D)
	for name, label in zip(model_names, labels):
		print(f"{name}: Cluster {label}")
	return labels

def plot_2d_embedding(D, labels, types, model_names, metric, method='mds'):
	"""Plot 2D embedding of diagrams using MDS, t-SNE, or UMAP, colored by cluster and type.
	
	Args:
		D: Distance matrix
		labels: Cluster labels
		types: Types for each sample
		model_names: Model names for each sample
		metric: Distance metric name (for title/filename)
		method: Embedding method - 'mds', 'tsne', or 'umap'
	"""
	if method == 'mds':
		embed = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
		coords = embed.fit_transform(D)
		method_label = 'MDS'
	elif method == 'tsne':
		embed = TSNE(n_components=2, metric="precomputed", init="random", random_state=42, perplexity=min(30, len(D)-1))
		coords = embed.fit_transform(D)
		method_label = 't-SNE'
	elif method == 'umap':
		if umap is None:
			print(f"Skipping UMAP (not installed)")
			return
		embed = umap.UMAP(n_components=2, metric="precomputed", random_state=42)
		coords = embed.fit_transform(D)
		method_label = 'UMAP'
	else:
		raise ValueError(f"Unknown method: {method}")
	
	plt.figure(figsize=(10,8))
	# Use color for model (0-6) and a unique color for PCA, all with the same marker
	import matplotlib.cm as cm
	import matplotlib.colors as mcolors
	# Clean model names: PCA is 'PCA', others are 0-6
	def clean_model(name, typ):
		if typ == "PCA":
			return "PCA"
		base = name.split('_batch')[0]
		if base.startswith("gmm_cnn_"):
			base = base.replace("gmm_cnn_", "")
		return base
	model_ids = [clean_model(n, t) for n, t in zip(model_names, types)]
	unique_models = []
	for m in model_ids:
		if m not in unique_models:
			unique_models.append(m)
	color_map = cm.get_cmap('tab20', len(unique_models))
	model_to_color = {model: mcolors.to_hex(color_map(i)) for i, model in enumerate(unique_models)}
	for xy, model in zip(coords, model_ids):
		color = model_to_color.get(model, '#000000')
		plt.scatter(xy[0], xy[1], c=color, marker='o', edgecolor="k", s=80, alpha=0.8)
	# Custom legend for models
	from matplotlib.lines import Line2D
	legend_elements = [
		Line2D([0], [0], marker='o', color='w', label=str(model), markerfacecolor=model_to_color[model], markersize=10, markeredgecolor='k')
		for model in unique_models
	]
	plt.legend(handles=legend_elements, loc='best', fontsize=10)
	# Why do clustering results look similar across methods/metrics?
	# - If the underlying data is well-separated or has a dominant structure, most clustering methods will find similar groupings.
	# - Wasserstein and bottleneck distances are both based on matching points in persistence diagrams, so they may yield similar results if diagrams are not very different in structure.
	# - Try using only a subset of homology classes, or combining them, or using different distance parameters for more variety.
	plt.title(f"2D {method_label} of Persistence Diagrams ({metric})")
	plt.xlabel(f"{method_label} 1")
	plt.ylabel(f"{method_label} 2")
	plt.tight_layout()
	# Save image
	fname = f"clustering_{metric}_{method}.png"
	plt.savefig(fname)
	print(f"Saved plot to {fname}")
	plt.show()


if __name__ == "__main__":
	# You can adjust start_indices, max_batches, n_jobs as needed
	diagrams, names, types = load_all_diagrams_from_cache()
	print(f"Loaded {len(diagrams)} diagrams.")
	if len(diagrams) == 0:
		print("No valid persistence diagrams found. Exiting.")
	else:
		print("Sample diagram names:", names[:min(5, len(names))])
		
		# Extract subset: all VAE from all models + all PCA from first model only
		model_names_clean = [n.split('_batch')[0] for n in names]
		unique_models = []
		for m in model_names_clean:
			if m not in unique_models:
				unique_models.append(m)
		
		print(f"Found {len(unique_models)} unique models: {unique_models}")
		
		# Extract all VAE indices for each model
		vae_indices_by_model = []
		for m in unique_models:
			vae_indices = [i for i, (mn, t) in enumerate(zip(model_names_clean, types)) if mn == m and t == "VAE"]
			vae_indices_by_model.append(vae_indices)
			print(f"Model {m}: {len(vae_indices)} VAE diagrams")
		
		# Extract all PCA indices from the FIRST model only (since PCA is the same for all models)
		pca_indices = [i for i, (mn, t) in enumerate(zip(model_names_clean, types)) if mn == unique_models[0] and t == "PCA"]
		print(f"PCA (from {unique_models[0]}): {len(pca_indices)} diagrams")
		
		# Build the subset: VAE0, VAE1, ..., VAE6, PCA
		selected_indices = []
		subset_names = []
		subset_types = []
		
		for m_idx, vi in enumerate(vae_indices_by_model):
			selected_indices.extend(vi)
			subset_names.extend([f"VAE{m_idx}"] * len(vi))
			subset_types.extend(["VAE"] * len(vi))
		
		selected_indices.extend(pca_indices)
		subset_names.extend(["PCA"] * len(pca_indices))
		subset_types.extend(["PCA"] * len(pca_indices))
		
		# Extract the subset of diagrams
		subset_diagrams = [diagrams[i] for i in selected_indices]
		print(f"\nSubset: {len(subset_diagrams)} diagrams total")
		print(f"Groups: {len(vae_indices_by_model)} VAE models + 1 PCA group")
		
		# Compute distance matrices for this subset only
		metric = "wasserstein"
		D_matrices = {}
		
		# STEP 1: Compute/load ALL distance matrices first (with caching)
		print("\n=== STEP 1: Computing/Loading Distance Matrices ===")
		for hom_dim in [0, 1, 2]:
			print(f"\nComputing {metric} distance matrix for subset, homology_dim={hom_dim}...")
			D = compute_distance_matrix(subset_diagrams, metric=metric, homology_dim=hom_dim)
			D_matrices[hom_dim] = D
		
		# Compute sum distance matrix
		if all(h in D_matrices for h in [0,1,2]):
			Dsum = D_matrices[0] + D_matrices[1] + D_matrices[2]
			D_matrices['sum'] = Dsum
			print(f"\nComputed sum distance matrix H0+H1+H2")
		
		# STEP 2: Now do all plotting
		print("\n=== STEP 2: Plotting ===")
		
		# Exclude VAE1, VAE2, and VAE5 by extracting submatrix
		print("\n=== Excluding VAE1, VAE2, and VAE5 from plots ===")
		n_per_group = len(vae_indices_by_model[0])
		n_models = len(vae_indices_by_model)
		
		# Create indices for all groups except VAE1, VAE2, and VAE5
		vae1_start = 1 * n_per_group
		vae1_end = 2 * n_per_group
		vae2_start = 2 * n_per_group
		vae2_end = 3 * n_per_group
		vae5_start = 5 * n_per_group
		vae5_end = 6 * n_per_group
		
		# Keep: VAE0, VAE3, VAE4, VAE6, PCA
		keep_indices = (list(range(vae1_start)) + 
		                list(range(vae2_end, vae5_start)) + 
		                list(range(vae5_end, len(subset_diagrams))))
		
		# Extract submatrices from distance matrices
		D_matrices_excl = {}
		for key, D in D_matrices.items():
			D_excl = D[np.ix_(keep_indices, keep_indices)]
			D_matrices_excl[key] = D_excl
			if isinstance(key, int):
				print(f"H{key}: Full matrix {D.shape} -> Excluded VAE1,2,5 matrix {D_excl.shape}")
		
		# Update subset_names and subset_types for excluded version
		subset_names_excl = [subset_names[i] for i in keep_indices]
		subset_types_excl = [subset_types[i] for i in keep_indices]
		
		# Relabel: VAE3->VAE1, VAE4->VAE2, VAE6->VAE3
		subset_names_excl = [name.replace("VAE3", "VAE_tmp3").replace("VAE4", "VAE_tmp4")
		                          .replace("VAE6", "VAE_tmp6")
		                          .replace("VAE_tmp3", "VAE1").replace("VAE_tmp4", "VAE2")
		                          .replace("VAE_tmp6", "VAE3")
		                      for name in subset_names_excl]
		
		# New group labels (4 VAE + 1 PCA = 5 groups)
		group_labels_excl = [f"VAE{i}" for i in range(n_models - 3)] + ["PCA"]
		n_groups_excl = len(group_labels_excl)
		
		print(f"After excluding VAE1, VAE2, and VAE5: {len(subset_names_excl)} diagrams, {n_groups_excl} groups")
		
		# Helper function for grouped heatmap
		def plot_grouped_heatmap(D, group_labels, n_per_group, title, fname):
			import matplotlib.pyplot as plt
			import numpy as np
			plt.figure(figsize=(12, 10))
			im = plt.imshow(D, aspect='auto', cmap='viridis')
			plt.colorbar(im, fraction=0.046, pad=0.04)
			n_groups = len(group_labels)
			tick_positions = [i * n_per_group + n_per_group // 2 for i in range(n_groups)]
			plt.xticks(tick_positions, group_labels, rotation=90, fontsize=12)
			plt.yticks(tick_positions, group_labels, fontsize=12)
			plt.title(title)
			plt.tight_layout()
			plt.savefig(fname)
			print(f"Saved grouped heatmap to {fname}")
			plt.close()
		
		for hom_dim in [0, 1, 2]:
			D_excl = D_matrices_excl[hom_dim]
			if D_excl.shape[0] < 2:
				print(f"Not enough diagrams for clustering H{hom_dim}. Skipping.")
				continue
			
			print(f"\nPlotting for H{hom_dim}...")
			# Plot grouped heatmap
			plot_grouped_heatmap(D_excl, group_labels_excl, n_per_group, 
				f"{metric} distance matrix H{hom_dim} (VAE0,1-3 + PCA, excl. orig VAE1,2,5)", 
				f"distance_heatmap_{metric}_h{hom_dim}.png")
			
			# Agglomerative clustering
			print(f"AgglomerativeClustering (average linkage) for H{hom_dim}...")
			labels_agg = cluster_and_report(D_excl, subset_names_excl, n_clusters=n_groups_excl, method="average")
			
			# Plot with all three embedding methods
			for embed_method in ['mds', 'tsne', 'umap']:
				print(f"  Creating {embed_method.upper()} embedding...")
				plot_2d_embedding(D_excl, labels_agg, subset_types_excl, subset_names_excl, 
								  f"{metric}_h{hom_dim}", method=embed_method)
		
		# Plot for sum H0+H1+H2
		if 'sum' in D_matrices_excl:
			Dsum_excl = D_matrices_excl['sum']
			print(f"\nPlotting for H0+H1+H2 sum...")
			
			# Plot grouped heatmap for sum
			plot_grouped_heatmap(Dsum_excl, group_labels_excl, n_per_group, 
				f"{metric} distance matrix H0+H1+H2 sum (VAE0,1-3 + PCA, excl. orig VAE1,2,5)", 
				f"distance_heatmap_{metric}_hsum.png")
			
			# Agglomerative clustering
			print("AgglomerativeClustering (average linkage) for sum...")
			labels_agg = cluster_and_report(Dsum_excl, subset_names_excl, n_clusters=n_groups_excl, method="average")
			
			# Plot with all three embedding methods
			for embed_method in ['mds', 'tsne', 'umap']:
				print(f"  Creating {embed_method.upper()} embedding...")
				plot_2d_embedding(Dsum_excl, labels_agg, subset_types_excl, subset_names_excl, 
								  f"{metric}_hsum", method=embed_method)
