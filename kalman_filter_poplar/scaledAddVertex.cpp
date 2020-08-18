#include <poplar/Vertex.hpp>

// M1 + s * M2 -> M3
// In principle s can be a matrix, but never is
// (Otherwise, merge with matrix product)

class ScaledAdd : public poplar::Vertex {
public:

  // Fields
  poplar::Vector<poplar::Input<poplar::Vector<float>>> first;
  poplar::Vector<poplar::Input<poplar::Vector<float>>> second;
  poplar::Input<float> s;
  poplar::Vector<poplar::Output<poplar::Vector<float>>> out;

  // Compute function
  bool compute() {

    int cols = first.size();
    int rows = first[0].size();

    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        out[i][j] = first[i][j] + s * second[i][j];
      }
    }

    return true;
  }

};

class Iterate : public poplar::Vertex {
public:

  // Fields
  poplar::Input<poplar::Vector<int>> in;
  poplar::Output<poplar::Vector<int>> out;

  // Compute function
  bool compute() {

    out[0] = in[0] + 1;

    return true;
  }

};
