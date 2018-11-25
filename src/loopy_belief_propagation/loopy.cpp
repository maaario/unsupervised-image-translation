#include <Python.h>
#include <vector>
#include <queue>
#include <cmath>
#include <iostream>

#include "dictionary_patches.cpp"
#include "io_matrix.cpp"
#include "latent_patches.cpp"
#include "message.cpp"

#define DEBUG false

using namespace std;

typedef long double ld;

// Structure implementing loopy belief propagation.
struct Loopy {
    // Number of iterations of message passing algorithm.
    int iterations;
    
    // Latent variables of MRF holding their prior distributions and messages.
    vector<LatentPatch> latent_patches;    
    // Dictionary of values that latent patches can achieve, dictionary patches
    // cnan be accesed by their id (position in the source image).
    vector<DictionaryPatch> dictionary_patches;
    
    ld two_sigma2;

    Loopy(ld two_sigma2): two_sigma2(two_sigma2) {}

    ld calculate_potential(ld distance) {
        return exp(-distance / two_sigma2);
    }

    // Create new message send from sender to receiver using the initial 
    // probabiliries of directory patches, past received messages and potential 
    // from overlaps.
    Message create_new_message(int sender, int receiver, int direction) {
        LatentPatch& lp_sender = latent_patches[sender];
        LatentPatch& lp_receiver = latent_patches[receiver];

        Message new_message(lp_receiver.k, 0);
        Message message_product = lp_sender.product_of_messages(direction);

        for (int j=0; j<lp_receiver.k; j++) {
            int dict_idx = lp_receiver.considered_dictionary_patches[j];
            DictionaryPatch& dp_receiver = dictionary_patches[dict_idx];
            
            for (int i=0; i<lp_sender.k; i++) {
                int dict_idx_2 = lp_receiver.considered_dictionary_patches[i];
                DictionaryPatch& dp_sender = dictionary_patches[dict_idx_2];
                ld potential = calculate_potential(
                    dp_sender.compute_overlap_distance(dp_receiver, direction));

                new_message.elements[j] += lp_sender.initial_probabilities[i] * 
                    potential * message_product.elements[i];
            }
            
        }

        new_message.normalize_sum();
        return new_message;
    }
    
    // Gnerates a random tree as a list of pairs (node, direction to neighbour).
    // The tree roughly correspondst to BFS search tree, but the next node
    // is selected randomly from the border of visited/unvisited nodes.
    vector<pair<int, int> > generate_random_tree() {
        vector<pair<int, int> > edges;

        int n = latent_patches.size();
        int first = rand() % n;
        vector<int> next_vertices(1, first);
        vector<bool> visited(n, false);
        visited[first] = true;

        while (next_vertices.size() > 0){
            int position = rand() % next_vertices.size();
            int current_vertex = next_vertices[position];
            next_vertices[position] = next_vertices.back();
            next_vertices.pop_back();

            for (int direction=0; direction<4; direction++) {
                int neighbour = 
                    latent_patches[current_vertex].neighbours[direction];
                if (neighbour == LatentPatch::NEIGHBOUR_UNREACHABLE || 
                    visited[neighbour]) {
                    continue;
                }
                next_vertices.push_back(neighbour);
                edges.push_back(make_pair(current_vertex, direction));
                visited[neighbour] = true;
            }
        }

        return edges;
    }

    // At first a random tree is genearated, messages are collected from leaves 
    // to the root and then messages are propagated from root to leaves.
    void execute_one_message_passing_iteration() {
        vector<pair<int, int> > edges = generate_random_tree();

        for (int i = int(edges.size()) - 1; i >= 0; i--) {
            int receiver = edges[i].first;
            int receiver_direction = edges[i].second;
            int sender = 
                latent_patches[receiver].neighbours[receiver_direction];
            int sender_direction = (receiver_direction + 2) % 4;
            latent_patches[receiver].received_messages[receiver_direction] = 
                create_new_message(sender, receiver, sender_direction);
        }

        for (int i=0; i<(int)edges.size(); i++) {
            int sender = edges[i].first;
            int sender_direction = edges[i].second;
            int receiver = latent_patches[sender].neighbours[sender_direction];
            int receiver_direction = (sender_direction + 2) % 4;
            latent_patches[receiver].received_messages[receiver_direction] = 
                create_new_message(sender, receiver, sender_direction);
        }
    }

    vector<vector<ld> > run_loopy(int iterations) {
        for(int iteration = 0; iteration < iterations; iteration++){
            execute_one_message_passing_iteration();
        }
        vector<vector<ld> > result_probabilities;
        for (int p=0; p<int(latent_patches.size()); p++) {
            result_probabilities.push_back(
                latent_patches[p].resulting_distribution());
        }
        return result_probabilities;
    }
};

extern "C" {
static PyObject *
loopy_belief_propagation(PyObject *self, PyObject *args)
{
    double two_sigma2;
    int iterations, grid_rows, grid_cols, patch_size, patch_overlap, k, seed;
    char * dictionary_vectors_path, * k_best_patches_path, 
         * k_best_probabilities_path, * result_probabilites_path;

    if (!PyArg_ParseTuple(args, "iiiiidiissss",
        &iterations, &grid_rows, &grid_cols, &patch_size, &patch_overlap, 
        &two_sigma2, &k, &seed, &dictionary_vectors_path, &k_best_patches_path, 
        &k_best_probabilities_path, &result_probabilites_path)) {
        return NULL;    
    }
    srandom(seed);
    
    vector<vector<int> > dictionary_vectors = 
        read_matrix_from_file<int>(
            dictionary_vectors_path, patch_size * patch_size);
    vector<vector<int> > k_best_patches = 
        read_matrix_from_file<int>(k_best_patches_path, k);
    vector<vector<ld> > k_best_probabilities = 
        read_matrix_from_file<ld>(k_best_probabilities_path, k);

    if (DEBUG) {
        cout << "Loopy loaded:\n" 
             << "    " << int(dictionary_vectors.size()) << " dictionary_vectors\n"
             << "    " << int(k_best_patches.size()) 
             << " lists of k best dict. patches for latent patches\n"
             << "    " << int(k_best_probabilities.size()) 
             << " lists of k best patch probabilities\n";
    }

    Loopy loopy(two_sigma2);
    loopy.dictionary_patches = prepare_dictionary_patches(
        dictionary_vectors, patch_size, patch_overlap);
    loopy.latent_patches = prepare_latent_patches(
        grid_rows, grid_cols, k_best_patches, k_best_probabilities);

    vector<vector<ld> > probabilities = loopy.run_loopy(iterations);
    write_matrix_to_file<ld>(result_probabilites_path, probabilities);

    return Py_BuildValue("i", 0);
}
}

static PyMethodDef LoopyMethods[] = {
    {"loopy_belief_propagation",  loopy_belief_propagation, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef loopy = {
   PyModuleDef_HEAD_INIT,
   "loopy",   /* name of module */
   NULL, /* module documentation, may be NULL */
   -1,       /* size of per-interpreter state of the module,
                or -1 if the module keeps state in global variables. */
   LoopyMethods
};

extern "C" {
PyMODINIT_FUNC
PyInit_loopy(void)
{
    return PyModule_Create(&loopy);
}
}

