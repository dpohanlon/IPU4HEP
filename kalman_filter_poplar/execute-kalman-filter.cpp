#include <iostream>
#include <vector>

#include <boost/timer/timer.hpp>

#include <poplar/DeviceManager.hpp>
#include <poplar/IPUModel.hpp>

#include <poplar/Engine.hpp>
#include <poplar/Program.hpp>
#include <poplin/codelets.hpp>
#include <popops/ElementWise.hpp>
#include <popops/codelets.hpp>
#include <popops/ExprOp.hpp>
#include <popops/Pad.hpp>
#include <poputil/TileMapping.hpp>

#include "KalmanFilter.h"

using namespace poplar;
using namespace poplar::program;
using namespace popops;

float d = 1.0;
float sigma = 10E-2;
int N = 5;
float z = 0.1;
float x0 = 0.01;
float theta0 = 10E-3;

int main(int argc, char const *argv[]) {

  Device dev = KalmanFilter::connectToIPU();

  // IPUModel ipuModel;
  // ipuModel.compileIPUCode = true;

  // Device dev = ipuModel.createDevice();

  Graph graph(dev.getTarget());

  popops::addCodelets(graph);
  poplin::addCodelets(graph);
  graph.addCodelets("matrixInverseVertex.cpp");
  graph.addCodelets("matrixProduct.cpp");
  graph.addCodelets("scaledAdd.cpp");
  graph.addCodelets("packHits.cpp");

  int n_inputs = 1;
  int batch_size = 1;

  std::vector<Tensor> inputs(n_inputs);
  std::vector<Tensor> inputs_batch(n_inputs);

  for (uint i = 0; i < inputs.size(); i++) {

    std::string iStr = std::to_string(i);

    inputs[i] = graph.addVariable(FLOAT, {5, 2}, "x_in" + iStr);
    inputs_batch[i] = graph.addVariable(FLOAT, {uint(batch_size), 5 * 2}, "x_in_batch" + iStr); // Check dims!
    graph.setTileMapping(inputs[i], i);
    graph.setTileMapping(inputs_batch[i], i);
  }

  Sequence preProg;

  std::vector<DataStream> inStreams(n_inputs);
  std::vector<Tensor> covs(n_inputs);
  std::vector<Tensor> qs(n_inputs);
  std::vector<Tensor> hs(n_inputs);
  std::vector<Tensor> gs(n_inputs);
  std::vector<Tensor> fs(n_inputs);
  std::vector<Tensor> d(n_inputs);
  std::vector<Tensor> dInit(n_inputs);
  std::vector<Tensor> dSkip(n_inputs);
  std::vector<Tensor> scatterInto(n_inputs);
  std::vector<Tensor> loop(n_inputs);
  std::vector<Tensor> zero(n_inputs);
  std::vector<Tensor> one(n_inputs);
  std::vector<Tensor> loop_batch(n_inputs);
  std::vector<Tensor> hitThisLoop(n_inputs);

  std::vector<Tensor> p_proj_all(n_inputs);
  std::vector<Tensor> C_proj_all(n_inputs);

  std::vector<Tensor> p_filt_all(n_inputs);
  std::vector<Tensor> C_filt_all(n_inputs);

  std::vector<Tensor> p_smooth(n_inputs);
  std::vector<Tensor> C_smooth(n_inputs);

  for (uint i = 0; i < covs.size(); i++) {

    std::string iStr = std::to_string(i);

    inStreams[i] = graph.addHostToDeviceFIFO("inStream" + iStr, FLOAT, 5 * 2 * batch_size);
    preProg.add(Copy(inStreams[i], inputs_batch[i]));

  }

  Sequence prog;

  std::vector<Tensor> covFlat(covs.size());

  for (uint i = 0; i < covs.size(); i++) {

    std::string iStr = std::to_string(i);

    loop[i] = graph.addVariable(INT, {1}, "loop");
    graph.setTileMapping(loop[i], i);

    scatterInto[i] = graph.addConstant<int>(INT, {1}, {0});
    graph.setTileMapping(scatterInto[i], i);

    zero[i] = graph.addConstant<int>(INT, {1}, {0});
    graph.setTileMapping(zero[i], i);

    one[i] = graph.addConstant<int>(INT, {1}, {1});
    graph.setTileMapping(one[i], i);

    prog.add(Copy(inputs_batch[i].slice(0, 1, 0).reshape({5, 2}), inputs[i]));

    covFlat[i] = graph.addConstant<float>(FLOAT, {16, 1}, {sigma * sigma, 0., 0., 0.,
                                                          0., M_PI, 0., 0.,
                                                          0., 0., sigma * sigma, 0.,
                                                          0., 0., 0., M_PI
                                                          });

    Tensor qFlat = graph.addConstant<float>(FLOAT, {16, 1}, {0.});

    Tensor hFlat = graph.addConstant<float>(FLOAT, {16, 1}, {1., 0., 0., 0.,
                                                             0., 0., 0., 0.,
                                                             0., 0., 1., 0.,
                                                             0., 0., 0., 0.});
    Tensor fFlat = graph.addConstant<float>(FLOAT, {16, 1}, {1., 1., 0., 0.,
                                                             0., 1., 0., 0.,
                                                             0., 0., 1., 1.,
                                                             0., 0., 0., 1.});
    Tensor gFlat = graph.addConstant<float>(FLOAT, {16, 1}, {float(1.0)/(sigma * sigma), 0., 0., 0.,
                                                             0, 0., 0., 0.,
                                                             0, 0., float(1.0)/(sigma * sigma), 0.,
                                                             0, 0., 0., 0.,
                                                             });

    d[i] = graph.addVariable(FLOAT, {1, 1}, "d" + iStr);
    dInit[i] = graph.addConstant<float>(FLOAT, {1, 1}, {1.});
    dSkip[i] = graph.addConstant<float>(FLOAT, {1, 1}, {2.});

    prog.add(Copy(dInit[i], d[i]));

    covs[i] = graph.addVariable(FLOAT, {4, 4}, "cov" + iStr);
    prog.add(Copy(covFlat[i].reshape({4, 4}), covs[i]));

    p_proj_all[i] = graph.addVariable(FLOAT, {5, 4, 1}, "p_proj_all" + iStr);
    C_proj_all[i] = graph.addVariable(FLOAT, {5, 4, 4}, "C_proj_all" + iStr);

    p_filt_all[i] = graph.addVariable(FLOAT, {5, 4, 1}, "p_filt_all" + iStr);
    C_filt_all[i] = graph.addVariable(FLOAT, {5, 4, 4}, "C_filt_all" + iStr);

    p_smooth[i] = graph.addVariable(FLOAT, {4, 1}, "p_smooth" + iStr);
    C_smooth[i] = graph.addVariable(FLOAT, {4, 4}, "C_smooth" + iStr);

    graph.setTileMapping(p_proj_all[i], i);
    graph.setTileMapping(C_proj_all[i], i);
    graph.setTileMapping(p_filt_all[i], i);
    graph.setTileMapping(C_filt_all[i], i);

    graph.setTileMapping(p_smooth[i], i);
    graph.setTileMapping(C_smooth[i], i);

    qs[i] = qFlat.reshape({4, 4});

    hs[i] = hFlat.reshape({4, 4});
    gs[i] = gFlat.reshape({4, 4});
    fs[i] = fFlat.reshape({4, 4});

    graph.setTileMapping(covFlat[i], i);
    graph.setTileMapping(qFlat, i);
    graph.setTileMapping(d[i], i);
    graph.setTileMapping(dInit[i], i);
    graph.setTileMapping(dSkip[i], i);

    graph.setTileMapping(covs[i], i);
    graph.setTileMapping(qs[i], i);

    graph.setTileMapping(gFlat, i);
    graph.setTileMapping(hFlat, i);
    graph.setTileMapping(fFlat, i);
    graph.setTileMapping(hs[i], i);
    graph.setTileMapping(gs[i], i);
    graph.setTileMapping(fs[i], i);
  }

  // Init p with hits
  auto [packIterationTensorsInit, ps] = KalmanFilter::packIterationTensors(graph, loop, scatterInto, inputs);

  prog.add(packIterationTensorsInit);

  // Prepare hits for each loop
  auto [packIterationTensorsProg, hits] = KalmanFilter::packIterationTensors(graph, loop, scatterInto, inputs);

  auto [projProg, p_proj, C_proj] = KalmanFilter::project(graph, ps, covs, fs, qs);

  auto [filterSeq, outP, C] = KalmanFilter::filter(graph, hs, gs, hits, p_proj, C_proj);

  std::vector<Tensor> p_proj_chi2(ps.size());
  std::vector<Tensor> p_filt_chi2(ps.size());
  std::vector<Tensor> C_proj_chi2(ps.size());

  std::vector<Tensor> chiSqThreshold(ps.size());

  // For chi2 calc, test
  for (uint i = 0; i < ps.size(); i++) {

    std::string iStr = std::to_string(i);

    p_proj_chi2[i] = graph.addVariable(FLOAT, {4, 1}, "p_proj_chi2" + iStr);
    graph.setTileMapping(p_proj_chi2[i], i);

    p_filt_chi2[i] = graph.addVariable(FLOAT, {4, 1}, "p_filt_chi2" + iStr);
    graph.setTileMapping(p_filt_chi2[i], i);

    C_proj_chi2[i] = graph.addVariable(FLOAT, {4, 4}, "C_proj_chi2" + iStr);
    graph.setTileMapping(C_proj_chi2[i], i);

    chiSqThreshold[i] = graph.addConstant<float>(FLOAT, {1}, {0.4});
    graph.setTileMapping(chiSqThreshold[i], i);

  }

  // ChiSq computation

  auto [resSeq, res] = KalmanFilter::calcResidual(graph, hits, p_filt_chi2, hs);

  auto [chiSqSeq, chiSq] = KalmanFilter::calcChiSq(graph, res, gs, C_proj_chi2, p_proj_chi2, p_filt_chi2);

  auto [chiSqTestSeq, chiSqTestPred] = KalmanFilter::chiSqTest(graph, chiSq, chiSqThreshold);

  // Update loop index
  Sequence updateIterator;

  for (uint i = 0; i < inputs.size(); i++) {

    auto [computeIterate, itr] = KalmanFilter::iterate(graph, loop[i], i);
    updateIterator.add(Execute(computeIterate));
    updateIterator.add(Copy(itr, loop[i]));

  }

  Sequence planeLoop;

  planeLoop.add(packIterationTensorsProg);

  // For EKF
  // planeLoop.add(stateProg);
  // planeLoop.add(jacProg);

  planeLoop.add(projProg);
  planeLoop.add(filterSeq);

  // Save proj and filt states for smoothing step

  for (uint i = 0; i < ps.size(); i++) {

    auto [append_p_proj_Seq, p_proj_new] = KalmanFilter::appendTo(graph, p_proj[i], loop[i], p_proj_all[i], i);
    auto [append_C_proj_Seq, C_proj_new] = KalmanFilter::appendTo(graph, C_proj[i], loop[i], C_proj_all[i], i);
    auto [append_p_filt_Seq, p_filt_new] = KalmanFilter::appendTo(graph, outP[i], loop[i], p_filt_all[i], i);
    auto [append_C_filt_Seq, C_filt_new] = KalmanFilter::appendTo(graph, C[i], loop[i], C_filt_all[i], i);

    planeLoop.add(Execute(append_p_proj_Seq));
    planeLoop.add(Execute(append_C_proj_Seq));
    planeLoop.add(Execute(append_p_filt_Seq));
    planeLoop.add(Execute(append_C_filt_Seq));

    planeLoop.add(Copy(p_proj_new, p_proj_all[i]));
    planeLoop.add(Copy(C_proj_new, C_proj_all[i]));
    planeLoop.add(Copy(p_filt_new, p_filt_all[i]));
    planeLoop.add(Copy(C_filt_new, C_filt_all[i]));

  }

  // For chi2 calc, test
  // for (uint i = 0; i < ps.size(); i++) {
  //
  //   planeLoop.add(Copy(p_proj[i], p_proj_chi2[i]));
  //   planeLoop.add(Copy(outP[i], p_filt_chi2[i]));
  //   planeLoop.add(Copy(C_proj[i], C_proj_chi2[i]));
  //
  // }

  // planeLoop.add(resSeq);
  // planeLoop.add(chiSqSeq);
  // planeLoop.add(chiSqTestSeq);

  planeLoop.add(updateIterator);

  // Set up p for next projection, using p_filt
  // Set up covs for next projection, using C(_filt)
  for (uint i = 0; i < inputs.size(); i++) {
    planeLoop.add(Copy(outP[i], ps[i]));
    planeLoop.add(Copy(C[i], covs[i]));
  }

  // for (uint i = 0; i < inputs.size(); i++) {
  //
  //   auto [skipSwitchSeq, switchP, switchC, outD] = skipSwitch(graph, ps[i], covs[i],
  //                                                     outP[i], C[i],
  //                                                     chiSq[i],
  //                                                     i);
  //   planeLoop.add(Execute(skipSwitchSeq));
  //   planeLoop.add(Copy(switchP, ps[i]));
  //   planeLoop.add(Copy(switchC, covs[i]));
  //   planeLoop.add(Copy(outD, d[i]));
  // }

  prog.add(poplar::program::Repeat(5, planeLoop));

  for (uint i = 0; i < inputs.size(); i++) {
    // Reset iterator for smoothing step
    prog.add(Copy(zero[i], loop[i]));

    // Copy last filtered state to initial smoothing state
    prog.add(Copy(outP[i], p_smooth[i]));
    prog.add(Copy(C[i], C_smooth[i]));
  }

  std::vector<Tensor> p_proj_smooth(inputs.size());
  std::vector<Tensor> C_proj_smooth(inputs.size());
  std::vector<Tensor> p_filt_smooth(inputs.size());
  std::vector<Tensor> C_filt_smooth(inputs.size());

  Sequence smoothLoop;

  for (uint i = 0; i < inputs.size(); i++) {

    // Offset iterator for filtered states (when starting from idx == 0)
    auto [loopFiltSeq, loopFilt] = KalmanFilter::iterate(graph, loop[i], i);
    smoothLoop.add(Execute(loopFiltSeq));

    auto [sm_p_proj_seq, p_proj] = KalmanFilter::smoothingState(graph, p_proj_all[i], loop[i], i);
    auto [sm_C_proj_seq, C_proj] = KalmanFilter::smoothingState(graph, C_proj_all[i], loop[i], i);
    auto [sm_p_filt_seq, p_filt] = KalmanFilter::smoothingState(graph, p_filt_all[i], loopFilt, i);
    auto [sm_C_filt_seq, C_filt] = KalmanFilter::smoothingState(graph, C_filt_all[i], loopFilt, i);

    smoothLoop.add(Execute(sm_p_proj_seq));
    smoothLoop.add(Execute(sm_C_proj_seq));
    smoothLoop.add(Execute(sm_p_filt_seq));
    smoothLoop.add(Execute(sm_C_filt_seq));

    p_proj_smooth[i] = p_proj;
    C_proj_smooth[i] = C_proj;
    p_filt_smooth[i] = p_filt;
    C_filt_smooth[i] = C_filt;

  }

  auto [smoothSeq, p_smooth_new, C_smooth_new] = KalmanFilter::smooth(graph, p_smooth, C_smooth, p_filt_smooth, C_filt_smooth, p_proj_smooth, C_proj_smooth, fs);

  smoothLoop.add(smoothSeq);

  smoothLoop.add(PrintTensor("p_smooth_new", p_smooth_new[0]));

  for (uint i = 0; i < inputs.size(); i++) {

    // Propagate last smoothed state
    smoothLoop.add(Copy(p_smooth_new[i], p_smooth[i]));
    smoothLoop.add(Copy(C_smooth_new[i], C_smooth[i]));

  }

  smoothLoop.add(updateIterator);

  prog.add(poplar::program::Repeat(4, smoothLoop));

  // After loop over planes, for the next in the batch:

  for (uint i = 0; i < inputs.size(); i++) {
    // Reset plane iterator for next batch
    prog.add(Copy(zero[i], loop[i]));
  }

  Sequence progRepeat;
  progRepeat.add(poplar::program::Repeat(batch_size, prog));

  Sequence progMain;
  progMain.add(preProg);
  progMain.add(progRepeat);

  // Engine engine(graph, prog);
  Engine engine(graph, progMain);
  engine.load(dev);

  // Test input
  std::vector<float> v1 = {
    -0.02062073, -0.12062073,
    -0.02062073, -0.22062073,
    -0.12062073, -0.42062073,
    -0.12062073, -0.52062073,
    -0.22062073, -0.62062073,
  };

  std::vector<std::vector<float>> vs;
  for (uint i = 0; i < inputs.size(); i++) {
    // Instead of 1 * 5 inputs, N * 5 inputs
    std::vector<float> v10;
    for (uint j = 0; j < batch_size; j++) {
      v10.insert(std::end(v10), std::begin(v1), std::end(v1));
    }
    vs.push_back(v10);
  }

  for (uint i = 0; i < inputs.size(); i++) {
    engine.connectStream(inStreams[i], &vs[i][0], &vs[i][5 * 2 * batch_size]);
  }

  {
  boost::timer::auto_cpu_timer t;
  engine.run(0);
  }

  return 0;
}
