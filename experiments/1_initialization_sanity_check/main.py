import os
import sys

from scipy.misc import imsave

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.join(PROJECT_DIR, "src", "unsupervised_image_translation"))
from experiment import prepare_argument_parser, set_up_experiment
from patches import plot_patch_vectors


if __name__ == "__main__":
    argparser = prepare_argument_parser()
    args, _ = argparser.parse_known_args()

    patches, em = set_up_experiment(args)

    # Check that splitting to patches works.
    observed_patches = plot_patch_vectors(patches.observed_vectors, patches.observed_grid_size, overlap=-2)
    imsave(os.path.join(args.output, "patches_input.png"), observed_patches)
    source_patches = plot_patch_vectors(patches.dictionary_vectors, patches.source_grid_size, overlap=-2)
    imsave(os.path.join(args.output, "patches_source.png"), source_patches)

    # Check that PCA works.
    input_pca_resconstruction = patches.pca.inverse_transform(patches.compact_observed_vectors)
    input_pca_image = plot_patch_vectors(input_pca_resconstruction, patches.observed_grid_size, patches.patch_overlap)
    imsave(os.path.join(args.output, "pca_input.png"), input_pca_image)
    source_pca_reconstruction = patches.pca.inverse_transform(patches.compact_dictionary_vectors)
    source_pca_image = plot_patch_vectors(source_pca_reconstruction, patches.source_grid_size, patches.patch_overlap)
    imsave(os.path.join(args.output, "pca_source.png"), source_pca_image)

    # Check that initialization works.
    imsave(os.path.join(args.output, "0_initial_posteriors.png"), em.MAP_image())
    print("Log a posterior:", em.log_a_posterior_probability())
    