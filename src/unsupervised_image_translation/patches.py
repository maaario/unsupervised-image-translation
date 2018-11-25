import numpy as np
from sklearn.decomposition import PCA

import utils


def image_to_patches(image, patch_size, overlap):
    """
    Splits an image to patches of patch_size x patch_size, with a given overlap.
    Incomplete patches (coming from right and bottom image border) are ignored.
    """
    step = patch_size - overlap
    height, width = np.shape(image)
    patches = []
    for y in range(0, height - patch_size + 1, step):
        for x in range(0, width - patch_size + 1, step):
            patches.append(image[y:(y + patch_size), x:(x + patch_size)])
    return np.array(patches)                  


def rows_cols_of_patches_in_image(image, patch_size, overlap):
    """
    Computes how many patches are created in each row and column from an image.
    """
    step = patch_size - overlap
    height, width = np.shape(image)
    return [(height - overlap) // step, (width - overlap) // step]


def patches_to_vectors(patches):
    """
    Flattens an array of 2D square patches into 1D vectors.
    """
    patch_count, patch_size, _ = patches.shape
    return patches.reshape(patch_count, patch_size * patch_size)


def plot_patch_vectors(vectors, grid_shape, overlap):
    """
    Transforms pixel vectors into square patches and plots them into 2D grid
    of grid_shape size with specified overlaps. If overlap value is negative,
    image patches are separated by black lines of thickness -overlap.
    """
    patch_size = round(vectors.shape[1] ** 0.5)
    patches_in_col, patches_in_row = grid_shape
    step = patch_size - overlap
    out_width = patches_in_row * step + overlap
    out_height = patches_in_col * step + overlap

    image = np.zeros([out_height, out_width])
    weights = np.zeros([out_height, out_width])

    p = 0
    for y in range(0, out_height - patch_size + 1, step):
        for x in range(0, out_width - patch_size + 1, step):
            patch = vectors[p].reshape([patch_size, patch_size])
            image[y:(y + patch_size), x:(x + patch_size)] += patch
            weights[y:(y + patch_size), x:(x + patch_size)] += 1
            p += 1
    
    if overlap > 0:
        image = np.divide(image, weights)

    return image  


class Patches:
    """
    Holds input and dictionary patches in form of vectors with all informations 
    needed to reconstruct images. Also stores compact patch representations
    in form of principal components.
    """
    def __init__(self, input_path, source_path, patch_size, patch_overlap, 
                pca_k, color):
        # Store scalar settings.    
        self.patch_size = patch_size
        self.vector_size = patch_size * patch_size
        self.patch_overlap = patch_overlap
        self.pca_k = pca_k
        self.color = color
        
        # Load images.
        if not color:
            self.source_image_contrast = utils.load_image(source_path)
            self.input_image_contrast = utils.load_image(input_path)
        else:
            input_image = utils.load_image_rgb(input_path)
            self.yiq_input_image = utils.rgb2yiq(input_image)
            self.input_image_contrast = self.yiq_input_image[:, :, 0] 
            
            source_image = utils.load_image_rgb(source_path)
            self.yiq_source_image = utils.rgb2yiq(source_image)
            self.source_image_contrast = self.yiq_source_image[:, :, 0] 
        
        # Create observed patches
        self.observed_vectors = patches_to_vectors(image_to_patches(
            self.input_image_contrast, patch_size, patch_overlap))
        self.observed_grid_size = rows_cols_of_patches_in_image(
            self.input_image_contrast, patch_size, patch_overlap)
        self.patch_count = self.observed_vectors.shape[0]
        
        # Creaate dictionary_patches
        self.dictionary_vectors = patches_to_vectors(image_to_patches(
            self.source_image_contrast, patch_size, patch_overlap))
        self.source_grid_size = rows_cols_of_patches_in_image(
            self.source_image_contrast, patch_size, patch_overlap)
        self.dictionary_size = self.dictionary_vectors.shape[0]

        # Create compact patches with PCA
        self.pca = PCA(n_components=pca_k)
        self.pca.fit(np.vstack([self.observed_vectors, 
                                self.dictionary_vectors]))
        self.compact_observed_vectors = self.pca.transform(
            self.observed_vectors)
        self.compact_dictionary_vectors = self.pca.transform(
            self.dictionary_vectors)
        
        # Print patch stats.
        print("Patch count (P)           :", self.patch_count)
        print("Input image split to grid :", self.observed_grid_size)
        print("-"*30)
        print("Dictionary size (T = |mu|):", self.dictionary_size)
        print("Source image split to grid:", self.source_grid_size)
        print("-"*30)
        print("Vector dimensionality reduction: {} -> {}".format(
            self.vector_size, self.pca_k))
        explained_variance =  np.sum(self.pca.explained_variance_ratio_)
        print("Variance explained by PC:", explained_variance)

    def reconstruct_image(self, most_probable_patches, 
                          reconstruct_in_color=None):
        """
        With an array of indices of most probable source patches for each 
        observed patch, creates an image by joining source patches. If yiq 
        original image is provided, an RGB image is returned where color 
        chennels are copied from original image.
        """
        reconsturcted_grayscale = plot_patch_vectors(
            vectors=self.dictionary_vectors[most_probable_patches, :], 
            grid_shape=self.observed_grid_size, 
            overlap=self.patch_overlap,
        )

        if reconstruct_in_color is None:
            reconstruct_in_color = self.color
        elif not self.color and reconstruct_in_color:
            print("WARNING: Grayscale input image can't be reconstructed in "
                  "color. It will be reconstructed as grayscale.")
            reconstruct_in_color = False

        if reconstruct_in_color:
            shape = reconsturcted_grayscale.shape
            color_reconstructed = self.yiq_input_image[:shape[0], :shape[1], :]
            color_reconstructed[:, :, 0] = reconsturcted_grayscale
            color_reconstructed = utils.yiq2rgb(color_reconstructed)
            return color_reconstructed
        else:
            return reconsturcted_grayscale
