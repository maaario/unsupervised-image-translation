import os
import sys

import numpy as np
from scipy.misc import imsave, toimage

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.join(PROJECT_DIR, "src", "unsupervised_image_translation"))
from experiment import prepare_argument_parser, set_up_experiment
from patches import plot_patch_vectors
from em import EM


def run_experiment(args):
    patches, em = set_up_experiment(args)

    imsave(os.path.join(args.output, "0_initial.png"), em.MAP_image())
    print("Initial log a posterior:", em.log_a_posterior_probability())
    
    idxs = np.random.randint(0, patches.patch_count, 20)
    observed_patches = patches.compact_observed_vectors[idxs, :]
    
    for i in range(0, args.em_iterations + 1):
        transformed = []
        
        most_probable_patches = em.find_most_probable_patches_from_k(
            np.max(em.probs, axis=-1)
        )
        source_patches = patches.compact_dictionary_vectors[
            most_probable_patches[idxs], :] 
            
        for j in range(args.num_transformations):
            most_probable_patches = em.find_most_probable_patches_from_k(em.probs[:, :, j])
            reconstructed = patches.reconstruct_image(most_probable_patches)
            imsave(os.path.join(args.output, "{}_iter_only_lambda_{}.png".format(i, j)), reconstructed)
            transformed.append(np.matmul(em.lambdas[j], source_patches.T).T)

        vectors = np.array([source_patches, *transformed, observed_patches])
        vectors = np.swapaxes(vectors, 0, 1).reshape([-1, patches.pca_k])
        vectors = patches.pca.inverse_transform(vectors)
        trans = plot_patch_vectors(vectors, [20, 2 + args.num_transformations], -3)
        imsave(os.path.join(args.output, "patch_transformations_iter_{}.png".format(i)), trans)
        toimage(trans, cmin=0, cmax=1).save(os.path.join(args.output, "clipped_patch_transformations_iter_{}.png".format(i)))
        
        if (i < args.em_iterations):
            print("Executing EM iteration", i+1)
            em.execute_iteration()
            imsave(os.path.join(args.output, "{}_iter_MAP_image.png".format(i+1)), em.MAP_image())
            print("Log a posterior:", em.log_a_posterior_probability())
    

if __name__ == "__main__":
    argparser = prepare_argument_parser()
    args, _ = argparser.parse_known_args()
    
    output = args.output

    for init_type in EM.lambdas_init_dict.keys():
        args.init_transformations = init_type
        args.output = output + "_" + init_type
        run_experiment(args)
    