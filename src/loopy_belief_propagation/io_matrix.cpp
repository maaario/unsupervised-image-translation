#ifndef CPP_IO_MATRIX
#define CPP_IO_MATRIX

#include <fstream>
#include <vector>

using namespace std;

// Read a matrix from text file to 2D vector. To avoid searching for newline 
// characters, define how many cols should be read. 
template<class T> 
vector<vector<T> > read_matrix_from_file(char * path, int cols) {
    ifstream file(path);
    vector<vector<T> > matrix;
    T buffer;
    while (!file.eof()) {
        if(matrix.empty() || int(matrix.back().size()) >= cols) {
            matrix.push_back(vector<T>(0));
        }
        file >> buffer;
        matrix.back().push_back(buffer);
    }
    if (int(matrix.back().size()) < cols) {
        matrix.pop_back();
    }
    return matrix;
}

// Write matrix to text file. Each row will be written as a space separated 
// line of values.
template<class T> 
void write_matrix_to_file(char * path, vector<vector<T> >& matrix) {
    ofstream file(path);
    for (int row = 0; row < int(matrix.size()); row++) {
        for (int col = 0; col < int(matrix[row].size()); col++) {
            file << matrix[row][col] << " ";
        }
        file << "\n";
    }
}

#endif