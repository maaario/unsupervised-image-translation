#ifndef CPP_MESSAGE
#define CPP_MESSAGE

#include <algorithm>
#include <vector>

typedef long double ld;

using namespace std;

// Message holds the signals delivered from sender node to receiver node.
// i-th element of the signal corresponds with how likely it is for 
// the receiver node to have i-th value, conditioned on the subgraph
// in direction of sender node.
struct Message {
    vector<ld> elements;

    Message(int length, ld values) {
        elements.resize(length, values);
    }

    Message(vector<ld> &elements) : elements(elements) {}

    // Normalizes the message so that its elements sum to 1.
    void normalize_sum(){
        ld sum = 0;
        for(int i=0; i<int(elements.size()); i++){
            sum += elements[i];
        }
        for(int i=0; i<int(elements.size()); i++){
            elements[i] /= sum;
        }
    }

    // Normalizes the message so that its elements are in range [0, 1].
    void normalize_max(){
        ld maxi = 0;
        for(int i=0; i<int(elements.size()); i++){
            maxi = max(maxi, elements[i]);
        }
        for(int i=0; i<int(elements.size()); i++){
            elements[i] /= maxi;
        }
    }

    // Poinntwise multiplies current message with another.
    void multiply(Message& other) {
        for(int i=0; i<int(elements.size()); i++){
            elements[i] *= other.elements[i];
        }
    }
};

#endif