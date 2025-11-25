import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

def sentsim_as_weighted_wordsim_cuda(wordsim, weight, idseqs, device=None):
    """
    Calculate sentence similarity based on weighted word similarity using PyTorch.
    
    Args:
        wordsim: (V, V) numpy array of word similarities
        weight: (V,) numpy array of word weights
        idseqs: List[List[int]], list of sentences where each sentence is a list of word indices
        device: torch device or int or str
    
    Returns:
        sim: (N, N) numpy array of sentence similarities
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Convert inputs to tensors
    # wordsim might be float64, convert to float32 for GPU efficiency usually
    wordsim_t = torch.as_tensor(wordsim, device=device, dtype=torch.float32)
    weight_t = torch.as_tensor(weight, device=device, dtype=torch.float32)
    
    n_sents = len(idseqs)
    vocab_size = wordsim.shape[0]
    
    # Create U matrix (N_sents, Vocab_size) representing normalized weighted word vectors for each sentence
    u = torch.zeros((n_sents, vocab_size), device=device, dtype=torch.float32)
    
    # Fill U matrix
    # Since idseqs is a list of lists with variable lengths, we iterate.
    # For very large N, this loop might be slow in Python, but typically N is a few thousands for benchmarks.
    for i, sent_indices in enumerate(idseqs):
        if not sent_indices:
            continue
            
        indices = torch.as_tensor(sent_indices, device=device, dtype=torch.long)
        w = weight_t[indices]
        
        # Add weights to the corresponding positions in U
        # scatter_add_ adds values to the tensor at specified indices
        u[i].scatter_add_(0, indices, w)
    
    # Normalize rows of U so that weights sum to 1 for each sentence
    row_sums = u.sum(dim=1, keepdim=True)
    # Avoid division by zero for empty sentences
    row_sums[row_sums == 0] = 1.0
    u = u / row_sums
    
    # Calculate Similarity Matrix M = U @ S @ U.T
    # where S is the word similarity matrix
    
    # First compute temp = U @ S
    temp = torch.matmul(u, wordsim_t)
    
    # Then compute M = temp @ U.T
    sim_matrix = torch.matmul(temp, u.t())
    
    return sim_matrix.cpu().numpy()



def sentsim_as_weighted_wordsim_cpu(wordsim, weight, idseqs):
    # Fallback to the same implementation, just force CPU
    return sentsim_as_weighted_wordsim_cuda(wordsim, weight, idseqs, device="cpu")

