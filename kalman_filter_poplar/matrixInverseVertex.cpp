#include <poplar/Vertex.hpp>

class MatrixInverse2 : public poplar::Vertex {
public:

  // Fields
  poplar::Vector<poplar::Input<poplar::Vector<float>>> in;
  poplar::Vector<poplar::Output<poplar::Vector<float>>> out;

  // Compute function
  bool compute() {

    float det = in[0][0] * in[1][1] - in[0][1] * in[1][0];

    if (det == 0) return false; // Matrix is singular

    out[0][0] =  in[1][1] * (1. / det);
    out[0][1] = -in[0][1] * (1. / det);
    out[1][0] = -in[1][0] * (1. / det);
    out[1][1] =  in[0][0] * (1. / det);

    return true;
  }
};

class MatrixInverse3 : public poplar::Vertex {
public:

  // Fields
  poplar::Vector<poplar::Input<poplar::Vector<float>>> in;
  poplar::Vector<poplar::Output<poplar::Vector<float>>> out;

  // Compute function
  bool compute() {

    float a = in[0][0];
    float b = in[0][1];
    float c = in[0][2];

    float d = in[1][0];
    float e = in[1][1];
    float f = in[1][2];

    float g = in[2][0];
    float h = in[2][1];
    float i = in[2][2];

    float A =  (e * i - f * h);
    float B = -(d * i - f * g);
    float C =  (d * h - e * g);

    float D = -(b * i - c * h);
    float E =  (a * i - c * g);
    float F = -(a * h - b * g);

    float G =  (b * f - c * e);
    float H = -(a * f - c * d);
    float I =  (a * e - b * d);

    float det = a * A + b * B + c * C;

    if (det == 0) return false; // Matrix is singular

    out[0][0] = A * (1. / det);
    out[0][1] = D * (1. / det);
    out[0][2] = G * (1. / det);

    out[1][0] = B * (1. / det);
    out[1][1] = E * (1. / det);
    out[1][2] = H * (1. / det);

    out[2][0] = C * (1. / det);
    out[2][1] = F * (1. / det);
    out[2][2] = I * (1. / det);

    return true;
  }
};

class MatrixInverse4 : public poplar::Vertex {
public:

  // Fields
  poplar::Vector<poplar::Input<poplar::Vector<float>>> in;
  poplar::Vector<poplar::Output<poplar::Vector<float>>> out;

  // Compute function
  bool compute() {

    // I copied this pretty much verbatim from StackOverflow:
    // https://stackoverflow.com/a/60374938

    float A2323 = in[2][2] * in[3][3] - in[2][3] * in[3][2];
    float A1323 = in[2][1] * in[3][3] - in[2][3] * in[3][1];
    float A1223 = in[2][1] * in[3][2] - in[2][2] * in[3][1];
    float A0323 = in[2][0] * in[3][3] - in[2][3] * in[3][0];
    float A0223 = in[2][0] * in[3][2] - in[2][2] * in[3][0];
    float A0123 = in[2][0] * in[3][1] - in[2][1] * in[3][0];
    float A2313 = in[1][2] * in[3][3] - in[1][3] * in[3][2];
    float A1313 = in[1][1] * in[3][3] - in[1][3] * in[3][1];
    float A1213 = in[1][1] * in[3][2] - in[1][2] * in[3][1];
    float A2312 = in[1][2] * in[2][3] - in[1][3] * in[2][2];
    float A1312 = in[1][1] * in[2][3] - in[1][3] * in[2][1];
    float A1212 = in[1][1] * in[2][2] - in[1][2] * in[2][1];
    float A0313 = in[1][0] * in[3][3] - in[1][3] * in[3][0];
    float A0213 = in[1][0] * in[3][2] - in[1][2] * in[3][0];
    float A0312 = in[1][0] * in[2][3] - in[1][3] * in[2][0];
    float A0212 = in[1][0] * in[2][2] - in[1][2] * in[2][0];
    float A0113 = in[1][0] * in[3][1] - in[1][1] * in[3][0];
    float A0112 = in[1][0] * in[2][1] - in[1][1] * in[2][0];

    float det = in[0][0] * ( in[1][1] * A2323 - in[1][2] * A1323 + in[1][3] * A1223 )
              - in[0][1] * ( in[1][0] * A2323 - in[1][2] * A0323 + in[1][3] * A0223 )
              + in[0][2] * ( in[1][0] * A1323 - in[1][1] * A0323 + in[1][3] * A0123 )
              - in[0][3] * ( in[1][0] * A1223 - in[1][1] * A0223 + in[1][2] * A0123 );
    det = 1 / det;

    if (det == 0) return false; // Matrix is singular

    out[0][0] = det *   ( in[1][1] * A2323 - in[1][2] * A1323 + in[1][3] * A1223 );
    out[0][1] = det * - ( in[0][1] * A2323 - in[0][2] * A1323 + in[0][3] * A1223 );
    out[0][2] = det *   ( in[0][1] * A2313 - in[0][2] * A1313 + in[0][3] * A1213 );
    out[0][3] = det * - ( in[0][1] * A2312 - in[0][2] * A1312 + in[0][3] * A1212 );
    out[1][0] = det * - ( in[1][0] * A2323 - in[1][2] * A0323 + in[1][3] * A0223 );
    out[1][1] = det *   ( in[0][0] * A2323 - in[0][2] * A0323 + in[0][3] * A0223 );
    out[1][2] = det * - ( in[0][0] * A2313 - in[0][2] * A0313 + in[0][3] * A0213 );
    out[1][3] = det *   ( in[0][0] * A2312 - in[0][2] * A0312 + in[0][3] * A0212 );
    out[2][0] = det *   ( in[1][0] * A1323 - in[1][1] * A0323 + in[1][3] * A0123 );
    out[2][1] = det * - ( in[0][0] * A1323 - in[0][1] * A0323 + in[0][3] * A0123 );
    out[2][2] = det *   ( in[0][0] * A1313 - in[0][1] * A0313 + in[0][3] * A0113 );
    out[2][3] = det * - ( in[0][0] * A1312 - in[0][1] * A0312 + in[0][3] * A0112 );
    out[3][0] = det * - ( in[1][0] * A1223 - in[1][1] * A0223 + in[1][2] * A0123 );
    out[3][1] = det *   ( in[0][0] * A1223 - in[0][1] * A0223 + in[0][2] * A0123 );
    out[3][2] = det * - ( in[0][0] * A1213 - in[0][1] * A0213 + in[0][2] * A0113 );
    out[3][3] = det *   ( in[0][0] * A1212 - in[0][1] * A0212 + in[0][2] * A0112 );

    return true;
  }
};
