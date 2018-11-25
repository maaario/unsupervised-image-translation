#ifndef CPP_LATENT_PATCHES
#define CPP_LATENT_PATCHES

#include <iostream>
#include <vector>

#include "message.cpp"

typedef long double ld;

using namespace std;

// A latent variable of the MRF holding its prior, messages and indices
// of neighbouring latent variables.
struct LatentPatch {
    // Count of considered dict. patches. 
    // Also a dimension of a received message.
    int k;

    // Indices and probabilities of k most probable dictionary patches.
    vector<int> considered_dictionary_patches;
    vector<ld> initial_probabilities;
    
    // List of lates received messages from each direction. 
    vector<Message> received_messages;
    
    // A constant denoting unreachable neigbours of borderline nodes.
    static const int NEIGHBOUR_UNREACHABLE = -1;
    // 4 indices of neighboring latent patches (or NEIGHBOUR_UNREACHABLE).
    vector<int> neighbours;

    LatentPatch(vector<int>& considered_dictionary_patches, 
                vector<ld>& initial_probabilities)
    : considered_dictionary_patches(considered_dictionary_patches), 
      initial_probabilities(initial_probabilities) {
        k = considered_dictionary_patches.size();
        received_messages.resize(4, Message(k, 1));
        neighbours.resize(4);
    }

    // Compute the poinwise product of all received messages with exception of
    // message at index "excluded_direction".
    Message product_of_messages(int excluded_direction) {
        Message result(k, 1);
        for (int i=0; i<int(received_messages.size()); i++) {
            if (i == excluded_direction) continue;
            result.multiply(received_messages[i]);
        }
        return result;
    }

    // Compute the distribution resulting from message passing.
    vector<ld> resulting_distribution() {
        Message product = product_of_messages(-1);
        Message initial(initial_probabilities);
        product.multiply(initial);
        product.normalize_sum();
        return product.elements;
    }
};

// Checks that the degree distibution is as expected.
void check_patch_graph(int rows, int cols, vector<LatentPatch> &patches) {
    vector<int> expected = {
        0, 
        0, 
        4, 
        2 * (rows - 2) + 2 * (cols - 2),
        (rows - 2) * (cols - 2)
    };

    vector<int> nodes_of_degree(5, 0);
    for (auto p : patches) {
        int n = 0;
        for (int i=0; i<4; i++)
            n += (p.neighbours[i] != LatentPatch::NEIGHBOUR_UNREACHABLE);
        nodes_of_degree[n]++;       
    } 
    for (int i=0; i<5; i++) {
        if (expected[i] != nodes_of_degree[i]) {
            cout << "ERROR: There should be " << expected[i] 
                 << " nodes of degree " << i << " but in the latent graph"
                 << " there are " << nodes_of_degree[i] << " nodes.\n";
        }
    }
}

// Creates separate patch objects from matrices of indices and probabilities.
// Connects patches by building the graph (setting patch neighbours).
vector<LatentPatch> prepare_latent_patches(int rows, int cols,
    vector<vector<int> > k_best_patches, 
    vector<vector<ld> > k_best_probabilities) {
    vector<LatentPatch> patches;
    for (int i=0; i<int(k_best_patches.size()); i++) {
        patches.emplace_back(k_best_patches[i], k_best_probabilities[i]);
    }

    int dx[4] = {0, 1, 0, -1};
    int dy[4] = {-1, 0, 1, 0};
    for (int i=0; i<rows; i++) {
        for (int j=0; j<cols; j++) {
            for (int direction=0; direction<4; direction++) {
                int i2 = i + dy[direction];
                int j2 = j + dx[direction];
                if (i2 < 0 || i2 >= rows || j2 < 0 || j2 >= cols) {
                    patches[i * cols + j].neighbours[direction] = 
                        LatentPatch::NEIGHBOUR_UNREACHABLE;    
                } else {
                    patches[i * cols + j].neighbours[direction] = 
                        i2 * cols + j2;
                }
            }
        }
    }

    // check_patch_graph(rows, cols, patches);

    return patches;
}

#endif