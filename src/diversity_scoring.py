import argparse
import json
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

# Conditional imports for heavy dependencies
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the diversity scoring script."""
    parser = argparse.ArgumentParser(
        description="Calculate diversity scores for Tree-of-Evolution samples"
    )
    parser.add_argument(
        "--input_path", 
        type=str,
        required=True,
        help="Path to the complexity scoring results JSON file"
    )
    parser.add_argument(
        "--output_path", 
        type=str,
        required=True,
        help="Path to save the diversity scoring results"
    )
    parser.add_argument(
        "--model_name", 
        type=str,
        default="Alibaba-NLP/gte-large-en-v1.5",
        help="Name of the sentence transformer model to use"
    )
    parser.add_argument(
        "--batch_size", 
        type=int,
        default=32,
        help="Batch size for embedding computation"
    )
    parser.add_argument(
        "--device", 
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use for computation (auto, cuda, mps, cpu). Default: auto"
    )

    return parser.parse_args()


def load_json(file_path: str) -> List[Dict[str, Any]]:
    """Load JSON data from file."""
    logger.info("Loading data from %s", file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info("Loaded %d samples", len(data))
    return data


def dump_json(data: List[Dict[str, Any]], file_path: str) -> None:
    """Save JSON data to file."""
    logger.info("Saving %d samples to %s", len(data), file_path)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Data saved to %s", file_path)


def parse_sample_id(sample_id: str) -> Tuple[str, int]:
    """
    Parse sample ID to extract root node and layer information.
    
    Args:
        sample_id: Sample ID in format like "1", "1_2", "1_2_3", etc.
        
    Returns:
        Tuple of (root_node, layer) where layer is 0-indexed
    """
    parts = sample_id.split('_')
    root_node = parts[0]
    layer = len(parts) - 1
    return root_node, layer


def group_samples_by_root(samples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group samples by their root node.
    
    Args:
        samples: List of sample dictionaries
        
    Returns:
        Dictionary mapping root node to list of samples
    """
    root_groups = {}

    for sample in samples:
        root_node, layer = parse_sample_id(sample["id"])
        sample["root_node"] = root_node
        sample["layer"] = layer

        if root_node not in root_groups:
            root_groups[root_node] = []
        root_groups[root_node].append(sample)

    logger.info("Grouped samples into %d root nodes", len(root_groups))
    return root_groups


def build_database(root_groups: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Build database by selecting the highest complexity sample from each root node.
    
    Args:
        root_groups: Dictionary mapping root node to list of samples
        
    Returns:
        List of database samples (one per root node)
    """
    database = []

    for root_node, samples in root_groups.items():
        # Find sample with highest complexity score
        best_sample = max(samples, key=lambda x: x.get("self complexity score", 0))
        database.append(best_sample)

    logger.info("Built database with %d samples", len(database))
    return database


def get_device(device_arg: str = "auto") -> str:
    """
    Get the best available device for computation.
    
    Args:
        device_arg: Device argument from command line ('auto', 'cuda', 'mps', 'cpu')
    
    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    if device_arg != "auto":
        if device_arg == "cuda" and TORCH_AVAILABLE and torch.cuda.is_available():
            logger.info("Using specified CUDA GPU for computation")
            return "cuda"
        elif device_arg == "mps" and TORCH_AVAILABLE and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("Using specified Apple MPS GPU for computation")
            return "mps"
        elif device_arg == "cpu":
            logger.info("Using specified CPU for computation")
            return "cpu"
        else:
            logger.warning("Specified device '%s' not available, falling back to auto detection", device_arg)

    # Auto detection
    if TORCH_AVAILABLE:
        if torch.cuda.is_available():
            device = 'cuda'
            logger.info("Auto-detected CUDA GPU for computation")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
            logger.info("Auto-detected Apple MPS GPU for computation")
        else:
            device = 'cpu'
            logger.info("Auto-detected CPU for computation")
    else:
        device = 'cpu'
        logger.info("PyTorch not available, using CPU for computation")
    
    return device


def compute_embeddings(texts: List[str], model_name: str, batch_size: int, device: str = "auto") -> np.ndarray:
    """
    Compute embeddings for a list of texts.
    
    Args:
        texts: List of text strings
        model_name: Name of the sentence transformer model
        batch_size: Batch size for processing
        device: Device to use for computation
        
    Returns:
        NumPy array of embeddings
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError("sentence-transformers is required for embedding computation. "
                         "Install with: pip install sentence-transformers")
    
    logger.info("Computing embeddings for %d texts using %s", len(texts), model_name)
    
    # Check if this is a GTE model that requires trust_remote_code
    if 'gte' in model_name.lower() or 'Alibaba-NLP' in model_name:
        model = SentenceTransformer(model_name, trust_remote_code=True)
    else:
        model = SentenceTransformer(model_name)

    actual_device = get_device(device)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, device=actual_device, convert_to_numpy=True)
    logger.info("Computed embeddings with shape %s", embeddings.shape)
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> 'faiss.Index':
    """
    Build FAISS index for similarity search.
    
    Args:
        embeddings: NumPy array of embeddings
        
    Returns:
        FAISS index
    """
    if not FAISS_AVAILABLE:
        raise ImportError("faiss is required for similarity search. "
                         "Install with: pip install faiss-cpu")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    logger.info("Built FAISS index with %d vectors", index.ntotal)
    return index


def calculate_diversity_scores(
    samples: List[Dict[str, Any]], 
    database: List[Dict[str, Any]], 
    model_name: str,
    batch_size: int,
    device: str = "auto"
) -> List[Dict[str, Any]]:
    """
    Calculate diversity scores for all samples.
    
    Args:
        samples: List of all samples
        database: List of database samples (one per root node)
        model_name: Name of the sentence transformer model
        batch_size: Batch size for processing
        device: Device to use for computation
        
    Returns:
        List of samples with diversity scores added
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE or not FAISS_AVAILABLE:
        raise ImportError("Both sentence-transformers and faiss are required for diversity scoring. "
                         "Install with: pip install sentence-transformers faiss-cpu")

    # Extract content for embedding computation
    all_contents = [sample.get("content", "") for sample in samples]
    db_contents = [sample.get("content", "") for sample in database]

    # Compute embeddings
    all_embeddings = compute_embeddings(all_contents, model_name, batch_size, device)
    db_embeddings = compute_embeddings(db_contents, model_name, batch_size, device)

    # Build FAISS index for database
    db_index = build_faiss_index(db_embeddings)

    # Calculate diversity scores
    results = []
    for i, sample in enumerate(tqdm(samples, desc="Calculating diversity scores")):
        sample_embedding = all_embeddings[i:i+1]  # Shape: (1, dimension)
        faiss.normalize_L2(sample_embedding)

        # Find top-10 most similar database samples
        similarities, indices = db_index.search(sample_embedding, 10)
        similarities = similarities[0]  # Shape: (10,)
        indices = indices[0]  # Shape: (10,)

        # Find the highest similarity with database samples from different root nodes
        sample_root = sample["root_node"]
        max_similarity = 0.0
        
        # Iterate through top-10 results to find first sample from different root
        for j in range(len(similarities)):
            db_idx = indices[j]
            similarity = similarities[j]
            db_sample = database[db_idx]
            if db_sample["root_node"] != sample_root:
                max_similarity = similarity
                break

        # Convert similarity to diversity score (1 - similarity)
        # Higher diversity score means more diverse (less similar)
        diversity_score = 1.0 - max_similarity

        # Add diversity score to sample
        sample_copy = sample.copy()
        sample_copy["self diversity score"] = float(diversity_score)
        results.append(sample_copy)

    logger.info("Completed diversity score calculation")
    return results


def main() -> None:
    """Main entry point for the diversity scoring script."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Check dependencies
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        logger.error("sentence-transformers is not installed. Install with: pip install sentence-transformers")
        return

    if not FAISS_AVAILABLE:
        logger.error("faiss is not installed. Install with: pip install faiss-cpu")
        return

    args = parse_arguments()

    # Load data
    data = load_json(args.input_path)

    # Group samples by root node
    root_groups = group_samples_by_root(data)

    # Build database (one sample per root node with highest complexity)
    database = build_database(root_groups)

    # Calculate diversity scores
    results = calculate_diversity_scores(
        data,
        database,
        args.model_name,
        args.batch_size,
        args.device
    )

    # Save results
    dump_json(results, args.output_path)

    logger.info("Diversity scoring completed successfully!")


if __name__ == "__main__":
    main()
