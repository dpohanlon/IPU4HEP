#include <poplar/Vertex.hpp>

class PackHits : public poplar::Vertex {
public:

  // Fields
  poplar::Vector<poplar::Input<poplar::Vector<float>>> inputHits;
  poplar::Input<poplar::Vector<int>> itr;
  poplar::Vector<poplar::Output<poplar::Vector<float>>> out;

  // Compute function
  bool compute() {

    int inCols = inputHits.size();
    int outCols = 2 * inCols;

    out[0][0] = inputHits[0][itr[0]];
    out[0][1] = 0;

    return true;
  }

};

class PackHits4 : public poplar::Vertex {
public:

  // Fields
  poplar::Vector<poplar::Input<poplar::Vector<float>>> inputHits;
  poplar::Input<poplar::Vector<int>> itr;
  poplar::Vector<poplar::Output<poplar::Vector<float>>> out;

  // Compute function
  bool compute() {

    int inCols = inputHits.size();
    int outCols = 2 * inCols;
    int i = itr[0];

    out[0][0] = inputHits[i][0];
    out[0][1] = 0;
    out[0][2] = inputHits[i][1];
    out[0][3] = 0;

    return true;
  }

};

class StateForSmoothing : public poplar::Vertex {
public:

  // Fields
  poplar::Input<poplar::Vector<int>> itr;
  poplar::Vector<poplar::Input<poplar::Vector<float>>> states;
  // Return flattened state
  poplar::Output<poplar::Vector<float>> state;

  // Compute function
  bool compute() {

    // states is (4, 1, N) or (4, 4, N)
    // state is the inner dimension, (4, 1) or (4, 4) at itr

    int n = itr[0];
    // Check me
    // int outRows = states.size();
    int outCols = states[0].size();

    for (int j = 0; j < outCols; j++) {
      state[j] = states[n][j];
    }

    return true;
  }

};

class AppendState : public poplar::Vertex {
public:

  // Fields
  poplar::Input<poplar::Vector<int>> itr;
  poplar::Input<poplar::Vector<float>> state;
  poplar::Vector<poplar::Input<poplar::Vector<float>>> states;
  // How to get 3D input tensor? Just flatten for now...
  // poplar::Vector<poplar::Vector<poplar::Output<poplar::Vector<float>>>> states;
  // poplar::Vector<poplar::Output<poplar::Vector<poplar::Vector<float>>>> states;
  poplar::Vector<poplar::Output<poplar::Vector<float>>> newStates;

  // Compute function
  bool compute() {

    // states is (4, 1, N) or (4, 4, N)
    // state is the inner dimension, (4, 1) or (4, 4) at itr

    int n = itr[0];
    // Check me
    int outRows = states.size();
    int outCols = states[0].size();

    int total = outRows - 1;

    // Copy states into newStates

    for (int i = 0; i < outRows; i++) {
      for (int j = 0; j < outCols; j++) {
        newStates[i][j] = states[i][j];
      }
    }

    // Fill back to front, so that smooth iterator can go forward to back
    for (int j = 0; j < outCols; j++) {
      newStates[total - n][j] = state[j];
    }

    return true;
  }

};


class SkipSwitch : public poplar::Vertex {
public:

  // Fields
  poplar::Vector<poplar::Input<poplar::Vector<float>>> inputPOld;
  poplar::Vector<poplar::Input<poplar::Vector<float>>> inputCOld;

  poplar::Vector<poplar::Input<poplar::Vector<float>>> inputPNew;
  poplar::Vector<poplar::Input<poplar::Vector<float>>> inputCNew;

  poplar::Input<poplar::Vector<float>> chiSq;

  poplar::Vector<poplar::Output<poplar::Vector<float>>> outP;
  poplar::Vector<poplar::Output<poplar::Vector<float>>> outC;

  poplar::Vector<poplar::Output<poplar::Vector<float>>> outD;

  void copyOut(poplar::Vector<poplar::Input<poplar::Vector<float>>> p,
               poplar::Vector<poplar::Input<poplar::Vector<float>>> C)
  {
    for (int i = 0; i < p.size(); i++) {
      outP[i] = p[i];
    }

    for (int i = 0; i < C.size(); i++) {
      for (int j = 0; j < C[0].size(); j++) {
        outC[i][j] = C[i][j];
      }
    }
  }

  bool compute() {

    float threshold = 0.5;

    if (chiSq[0] > threshold) {
      copyOut(inputPOld, inputCOld);
      outD[0][0] = 2.0;
    } else {
      copyOut(inputPNew, inputCNew);
      outD[0][0] = 1.0;
    }

    return true;
  }

};
