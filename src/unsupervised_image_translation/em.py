import os

import numpy as np
from scipy.stats import multivariate_normal

import loopy
import utils


def loopy_belief_propagation(patches, k_indices, k_posteriors, lbp_params):
    """
    A wrapper function handling loopy belief propagation, mostly parameter 
    passing and communication via matrices in text files.
    """
    two_sigma2 = lbp_params["two_sigma2"]
    output_dir = lbp_params["output_dir"]
    iterations = lbp_params["iterations"]
    seed = lbp_params["seed"]

    io_file_paths = [
        "dictionary_vectors.txt",
        "k_best_patch_indices.txt",
        "k_best_probabilities.txt",
        "result_probabilities.txt",
    ]
    
    io_file_paths = [
        os.path.join(output_dir, filename) for filename in io_file_paths
    ]
     
    np.savetxt(io_file_paths[0], patches.dictionary_vectors * 255, fmt='%i')
    np.savetxt(io_file_paths[1], k_indices, fmt='%i')
    np.savetxt(io_file_paths[2], k_posteriors)

    patches_in_row, patches_in_col = patches.observed_grid_size
    patch_size = patches.patch_size
    patch_overlap = patches.patch_overlap
    k = k_indices.shape[1]

    loopy.loopy_belief_propagation(
        iterations, patches_in_row, patches_in_col, patch_size, patch_overlap, 
        two_sigma2, k, seed, *io_file_paths
    )
   
    result_probabilites = np.loadtxt(io_file_paths[3])
    
    return result_probabilites


class EM:
    def __init__(self, patches, num_candidates, num_transformations, 
                 lbp_params, lambdas_init_type):
        self.patches = patches
        self.num_candidates = num_candidates
        self.num_transformations = num_transformations
        self.lbp_params = lbp_params

        self.initial_posteriors = \
            self.compute_initial_posterior_marginals(patches)
        
        # Calculate initial P(y, t) (prop. to P(y | t)) to select relevant candidates.
        self.candidate_indices, init_probs = \
            self.find_k_most_probable_patches(self.initial_posteriors)
        init_probs = init_probs / np.sum(init_probs, axis=1, keepdims=True)
        
        # Set initial P(l) as random.
        init_lambdas_marginals = np.random.rand(self.patches.patch_count, num_transformations)
        init_lambdas_marginals /= np.sum(init_probs, axis=-1, keepdims=True)

        # Set initial P(t) as uniform
        self.loopy_probs = np.ones([self.patches.patch_count, self.num_candidates])
        self.loopy_probs /= self.num_candidates

        # Set initial P(y, t, l) = P(y, t) * P(l)
        self.probs = np.array([
                np.outer(init_probs[p], init_lambdas_marginals[p])
                for p in range(self.patches.patch_count)
            ]).reshape([patches.patch_count, num_candidates, num_transformations])
        
        # Set initial lambdas and psis.
        self.lambdas = np.array([
            self.lambdas_init_dict[lambdas_init_type](self.patches.pca_k)
            for _ in range(self.num_transformations)
        ])
        self.recompute_psis()

        
    def compute_initial_posterior_marginals(self, patches):
        """
        Returns a 2D array where posterior[observed_patch][dictionary_patch]
        is probability of generating observed patch x_i from dictionary patch.
        """
        covmat = np.identity(patches.pca_k)

        posteriors = np.zeros([patches.patch_count, patches.dictionary_size])
        for t in range(patches.dictionary_size):
            posteriors[:, t] = multivariate_normal.pdf(
                patches.compact_observed_vectors, 
                mean=patches.compact_dictionary_vectors[t], 
                cov=covmat
            )

        posteriors /= np.sum(posteriors, axis=1, keepdims=True)

        return posteriors

    def find_k_most_probable_patches(self, probabilities):
        """
        Returns a 2D array of indices: indices of  k most probable dictionary
        patches for each observed patch.
        Returns a 2D array of probabilities: probabilities of k most probable 
        dictionary patches for each observed patch.
        """
        sorted_probabilities = probabilities.copy()
        sorted_probabilities.sort()
        
        k_indices = probabilities.argsort()[:, -self.num_candidates:]
        k_posteriors = sorted_probabilities[:, -self.num_candidates:]
        return k_indices, k_posteriors

    def find_most_probable_patches_from_k(self, k_probabilities):
        """
        Returns an array of indices: index of most probable dictionary patch 
        for each observed patch.
        """
        return self.candidate_indices[
            np.arange(self.patches.patch_count), k_probabilities.argmax(axis=-1)
        ]

    def MAP_loopy_image(self):
        most_probable_patches = self.find_most_probable_patches_from_k(
            self.loopy_probs)
        return self.patches.reconstruct_image(most_probable_patches)

    def MAP_image(self):
        most_probable_patches = self.find_most_probable_patches_from_k(
            np.max(self.probs, axis=-1))
        return self.patches.reconstruct_image(most_probable_patches)
    
    def log_a_posterior_probability(self):
        """
        This probability does not contain MRF potentials, but only uses
        loopy posteriors as priors for selected dictionary patches.

        """
        return np.sum(np.log(np.max(self.probs, axis=(1, 2))))
        
    def loopy(self):
        """
        Runs loopy belief propagation on probabilities with marginalized 
        transformations.
        """
        self.loopy_probs = loopy_belief_propagation(
            patches=self.patches, 
            k_indices=self.candidate_indices, 
            k_posteriors=np.sum(self.probs, axis=-1), 
            lbp_params=self.lbp_params,
        )
    
    def unnormalized_multivariate_normal_pdf(self, x, mean, invcov):
        """
        Optimized version.
        As we normalize P(l_p, t_p | y_p) for each patch seaparately,
        we don't need to compute the normalization factor in normal PDF, 
        only renormalize all probabilities to sum to 1.
        Also, as we compite pdfs more times with one covariance matrix,
        we first compute its inverse and then reuse it.
        """
        diff = (x - mean)
        diff_cov_diff = np.clip(np.dot(np.dot(diff, invcov), diff), -100, 100)
        return np.exp(- 0.5 * diff_cov_diff)

    def compute_posteriors(self):
        marginals_for_lambdas = np.sum(self.probs, axis=1)
        
        new_probs = np.zeros(self.probs.shape)

        for p in range(self.patches.patch_count):
            invpsi = np.linalg.pinv(self.psis[p])
            for t in range(self.num_candidates):
                patch = self.patches.compact_dictionary_vectors[
                    self.candidate_indices[p][t]]
                for l in range(self.num_transformations):
                    transformed_patch = np.matmul(self.lambdas[l], patch)
                    new_probs[p][t][l] = (
                        self.loopy_probs[p][t] * 
                        marginals_for_lambdas[p][l] * 
                        self.unnormalized_multivariate_normal_pdf(
                            self.patches.compact_observed_vectors[p], 
                            mean=transformed_patch, 
                            invcov=invpsi,
                        )
                    )  
        
        self.probs = new_probs / np.sum(new_probs, axis=(1, 2), keepdims=True)      
    
    def recompute_lambdas(self):
        self.lambdas = np.zeros(self.lambdas.shape)

        for l in range(self.num_transformations):
            lambda_denominator = np.zeros(self.lambdas[l].shape)
                        
            for p in range(self.patches.patch_count):
                candidates = self.patches.compact_dictionary_vectors[
                    self.candidate_indices[p]]
                for t in range(self.num_candidates):
                    self.lambdas[l] += ( 
                        self.probs[p, t, l] * 
                        np.outer(self.patches.compact_observed_vectors[p], 
                                 candidates[t])
                    )

                    lambda_denominator += (
                        self.probs[p, t, l] * 
                        np.outer(candidates[t], candidates[t])
                    )

        lambda_denominator = np.linalg.inv(lambda_denominator)

        for l in range(self.num_transformations):
            self.lambdas[l] = self.lambdas[l].dot(lambda_denominator)

    def recompute_psis(self):  
        self.psis = np.zeros([self.patches.patch_count, 
                              self.patches.pca_k, self.patches.pca_k])
            
        for p in range(self.patches.patch_count):
            candidates = self.patches.compact_dictionary_vectors[
                    self.candidate_indices[p]]
            for t in range(self.num_candidates):
                for l in range(self.num_transformations):
                    diff = (
                        self.patches.compact_observed_vectors[p] -
                        np.matmul(self.lambdas[l], candidates[t])
                    )
                    self.psis[p] += ( 
                        self.probs[p, t, l] * np.outer(diff, diff)
                    )

        self.psis /= np.sum(self.probs, axis=(1, 2), keepdims=True)
    
    def maximization(self):
        self.recompute_lambdas()
        self.recompute_psis()

    def execute_iteration(self):
        self.loopy()
        self.compute_posteriors()
        self.maximization()
        self.compute_posteriors()

EM.lambdas_init_dict = {
   "id": 
        lambda k: np.identity(k),
    "rand":
        lambda k: np.random.rand(k, k),
    "rand_diag":
        lambda k: np.identity(k) * (np.random.rand(k) - 0.5), 
    "noisy_id":
        lambda k: np.identity(k) + (0.2 * np.random.rand(k, k) - 0.1),
}
