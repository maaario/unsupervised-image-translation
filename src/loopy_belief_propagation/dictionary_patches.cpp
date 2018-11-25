#ifndef CPP_DICTIONARY_PATCHES
#define CPP_DICTIONARY_PATCHES

#include <iostream>
#include <vector>

using namespace std;

typedef long double ld;

// Struct that holds all source patch data necessary to compute potentials 
// in pairwise MRF: pixel values of the top, right, bottom and left overlapping
//  part of a patch.
struct DictionaryPatch {
    vector<vector<int> > overlapping_region_pixels;

    ld pixel_distance(int p1, int p2) {
        // Range: 0 - 1
        return ld((p1 - p2) * (p1 - p2)) / (255 * 255);
    }

    // Computes sum of pixel distances of overlapping pixels of current patch 
    // and other patch where second patch is in direction of 
    // "relative_direction" of the current patch.
    ld compute_overlap_distance(DictionaryPatch &other, 
                                int relative_direction) {
        int other_patch_direction = (relative_direction + 2) % 4;
        ld distance_sum = 0;
        int overlapping_pixels_count = overlapping_region_pixels[0].size();
        for (int i=0; i<overlapping_pixels_count; i++) {
            distance_sum += pixel_distance(
                overlapping_region_pixels[relative_direction][i],
                other.overlapping_region_pixels[other_patch_direction][i]
            );
        }
        return distance_sum / overlapping_pixels_count;
    }
};

// Check that the overlapping regions from the source file have distances 0
void test_overlap_distances(vector<DictionaryPatch> &dictionary_patches) {
    int grid_rows = 29, grid_cols = 37;
    int dx[4] = {0, 1, 0, -1}, dy[4] = {-1, 0, 1, 0};
    for (int i=0; i<grid_rows; i++) for(int j=0; j<grid_cols; j++) 
        for (int direction = 0; direction < 4; direction++) {
        int i2 = i + dy[direction];
        int j2 = j + dx[direction];
        if (i2 < 0 || i2 >= grid_rows || j2 < 0 || j2 >= grid_cols) continue;
        DictionaryPatch &first = dictionary_patches[i * grid_cols + j];
        DictionaryPatch &second = dictionary_patches[i2 * grid_cols + j2];
        ld check_distance = first.compute_overlap_distance(second, direction);
        if (check_distance != 0) {
            cout << "ERROR: patches on positions (" 
                 << i << ", " << j << ") and ("
                 << i2 << ", " << j2 << ") should share overlap in direction " 
                 << direction
                 << " and their overlap distance should be 0, but it is " 
                 << check_distance << "\n";
        }
    } 
}

// Converts list of dictionary patch vectors into a list of DictionaryPatch.
vector<DictionaryPatch> prepare_dictionary_patches(
    vector<vector<int> > &patch_vectors, int patch_size, int patch_overlap) {
    int y0_for_direction[4] = {0, 0, patch_size - patch_overlap, 0};
    int dy_for_direction[4] = {
        patch_overlap, patch_size, patch_overlap, patch_size};
    int x0_for_direction[4] = {0, patch_size - patch_overlap, 0, 0};
    int dx_for_direction[4] = {
        patch_size, patch_overlap, patch_size, patch_overlap};

    vector<DictionaryPatch> dictionary_patches;

    for (int id = 0; id < int(patch_vectors.size()); id++) {
        DictionaryPatch patch;
        
        // Converts a vector of pixels int 2D vector of pixel values.
        vector<vector<int> > pixel_values(patch_size, vector<int>(patch_size));
        for (int i=0; i<patch_size; i++) {
            for (int j=0; j<patch_size; j++) {
                pixel_values[i][j] = patch_vectors[id][i * patch_size + j];
            }
        }

        // Computes subsets of pixels in part of patch given by direction
        // 0 ... 4 = top, right, bottom, left.
        patch.overlapping_region_pixels.resize(4);
        for (int direction = 0; direction < 4; direction++) {
            int y0 = y0_for_direction[direction];
            int ymax = y0 + dy_for_direction[direction];
            int x0 = x0_for_direction[direction];
            int xmax = x0 + dx_for_direction[direction];
            for (int y = y0; y < ymax; y++) {
                for (int x = x0; x < xmax; x++) {
                    patch.overlapping_region_pixels[direction].push_back(
                        pixel_values[y][x]
                    );
                }
            }
        }

        dictionary_patches.push_back(patch);        
    }

    // test_overlap_distances(dictionary_patches);

    return dictionary_patches;
}

#endif