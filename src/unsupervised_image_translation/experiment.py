import argparse
import os
import pprint

import numpy as np
from scipy.misc import imsave

from patches import Patches
from em import EM


def prepare_argument_parser():
    argparser = argparse.ArgumentParser(description="Argparser for unsupervised image translation")
    
    argparser.add_argument("-source", default=None, 
        help="Path to source image (style).", required=True)
    argparser.add_argument("-input", default=None, 
        help="Path to input image (content).", required=True)
    argparser.add_argument("-output", default="output", 
        help="Path to output folder.")
    argparser.add_argument("-color", default="gray", 
        help="Colorscheme for output image.", choices=["gray", "color"])
    
    argparser.add_argument("-patch_size", type=int, default=15, 
        help="Input and source images will be split to patch_size x patch_size squares.")
    argparser.add_argument("-patch_overlap", type=int, default=3, 
        help="Overlap depth of two neighbouring patches.")
    argparser.add_argument("-pca_k", type=int, default=50, 
        help="Number of PCA componetns of patches.")
    
    argparser.add_argument("-lbp_iterations", type=int, default=8, 
        help="Number of iterations in loopy belief propagation.")
    argparser.add_argument("-lbp_two_sigma2", type=float, default=0.1, 
        help="Sigma in potential pairwise funcion. Local smoothness should increase with lower values.")
    
    argparser.add_argument("-num_candidates", type=int, default=16, 
        help="Number of most probable dictionary patches considered for each latent patch.")
    argparser.add_argument("-num_transformations", type=int, default=3, 
        help="Number of possible linear patch translations (L).")
    argparser.add_argument("-init_transformations", default="rand", 
        help="Type of transformation initialization.", choices=EM.lambdas_init_dict.keys())
    argparser.add_argument("-em_iterations", type=int, default=5, 
            help="Number of EM iterations.")


    argparser.add_argument("-random_seed", type=int, default=0, 
        help="Seed for random number generators.")
    
    return argparser


def set_up_experiment(args):
    """
    Create exp. directory if needed, store used arguments, create patches and em objects.
    """
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    with open(os.path.join(args.output, "args.txt"), "w") as f: 
        args_dict = vars(args)
        for k, v in args_dict.items():
            if isinstance(v, bool) and not v:
                continue
            f.write("-{}={} \\\n".format(k, v))

    np.random.seed(args.random_seed)

    patches = Patches(
        input_path=args.input, 
        source_path=args.source, 
        patch_size=args.patch_size, 
        patch_overlap=args.patch_overlap,
        pca_k=args.pca_k,
        color=(args.color == "color"),
    )

    initial_posteriors_path = os.path.join(args.output, "initial_posteriors.npy")
    
    lbp_params = dict()
    lbp_params["output_dir"] = args.output
    lbp_params["two_sigma2"] = args.lbp_two_sigma2
    lbp_params["iterations"] = args.lbp_iterations
    lbp_params["seed"] = args.random_seed

    em = EM(
        patches=patches, 
        num_candidates=args.num_candidates, 
        num_transformations=args.num_transformations, 
        lbp_params=lbp_params, 
        lambdas_init_type=args.init_transformations,
    )    

    return patches, em
