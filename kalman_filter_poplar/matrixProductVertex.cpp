#include <poplar/Vertex.hpp>

class MatrixProduct : public poplar::Vertex {
public:

  // Fields
  poplar::Vector<poplar::Input<poplar::Vector<float>>> first;
  poplar::Vector<poplar::Input<poplar::Vector<float>>> second;
  poplar::Vector<poplar::Output<poplar::Vector<float>>> out;

  // Compute function
  bool compute() {

    int rows1 = first.size();
    int cols1 = first[0].size();
    int cols2 = second[0].size();

    for (int i = 0; i < rows1; i++) {
      for (int j = 0; j < cols2; j++) {
        out[i][j] = 0;
      }
    }

    for (int i = 0; i < rows1; i++) {
      for (int j = 0; j < cols2; j++) {
        for (int k = 0; k < cols1; k++) {
          out[i][j] += first[i][k] * second[k][j];
        }
      }
    }

    return true;
  }

};
