// export POPLAR_ENGINE_OPTIONS='{"target.workerStackSizeInBytes":1024}'

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

Device KalmanFilter::connectToIPU()
{

  DeviceManager manager = DeviceManager::createDeviceManager();
  Device dev;

  // Attempt to connect to a single IPU
  bool success = false;
  for (auto &d : manager.getDevices(poplar::TargetType::IPU, 1)) {
    dev = std::move(d);
    std::cerr << "Trying to attach to IPU " << dev.getId();
    if ((success = dev.attach())) {
      std::cerr << " - attached" << std::endl;
      break;
    } else {
      std::cerr << std::endl;
    }
  }
  if (!success) {
    std::cerr << "Error attaching to device" << std::endl;
    exit(1);

  }

  return dev;
}

std::tuple<ComputeSet, Tensor, Tensor, Tensor> KalmanFilter::skipSwitch(Graph & graph,
                                                  const Tensor & inputPOld,
                                                  const Tensor & inputCOld,
                                                  const Tensor & inputPNew,
                                                  const Tensor & inputCNew,
                                                  const Tensor & chiSq,
                                                  uint tile)
{
  ComputeSet computeSet = graph.addComputeSet("SkipSwitch" + std::to_string(tile));
  VertexRef vtx = graph.addVertex(computeSet, "SkipSwitch");
  graph.setCycleEstimate(vtx, 100);

  Tensor outP = graph.addVariable(FLOAT, {2, 1}, "switchOutP" + std::to_string(tile));
  Tensor outC = graph.addVariable(FLOAT, {2, 2}, "switchOutC" + std::to_string(tile));
  Tensor outD = graph.addVariable(FLOAT, {1, 1}, "switchOutD" + std::to_string(tile));

  graph.connect(vtx["inputPOld"], inputPOld);
  graph.connect(vtx["inputCOld"], inputCOld);
  graph.connect(vtx["inputPNew"], inputPNew);
  graph.connect(vtx["inputCNew"], inputCNew);
  graph.connect(vtx["chiSq"], chiSq);

  graph.connect(vtx["outP"], outP);
  graph.connect(vtx["outC"], outC);
  graph.connect(vtx["outD"], outD);

  graph.setTileMapping(vtx, tile);
  graph.setTileMapping(outP, tile);
  graph.setTileMapping(outC, tile);
  graph.setTileMapping(outD, tile);

  return std::make_tuple(computeSet, outP, outC, outD);
}

std::tuple<ComputeSet, Tensor> KalmanFilter::smoothingState(Graph & graph, const Tensor & states, const Tensor & itr, uint tile)
{
  ComputeSet computeSet = graph.addComputeSet("smoothingState" + std::to_string(tile));
  VertexRef vtx = graph.addVertex(computeSet, "StateForSmoothing");
  graph.setCycleEstimate(vtx, 100);

  Tensor tmp = states.reshape({states.shape()[0], states.shape()[1] * states.shape()[2]});
  Tensor state = graph.addVariable(FLOAT, {states.shape()[1], states.shape()[2]}, "state" + std::to_string(tile));

  graph.connect(vtx["state"], state.reshape({state.shape()[0] * state.shape()[1]}));
  graph.connect(vtx["states"], tmp);
  graph.connect(vtx["itr"], itr);

  graph.setTileMapping(vtx, tile);
  graph.setTileMapping(tmp, tile);
  graph.setTileMapping(state, tile);

  return std::make_pair(computeSet, state.reshape({states.shape()[1], states.shape()[2]}));
}

std::tuple<ComputeSet, Tensor> KalmanFilter::appendTo(Graph & graph, const Tensor & state, const Tensor & itr, Tensor & states, uint tile)
{
  ComputeSet computeSet = graph.addComputeSet("append" + std::to_string(tile));
  VertexRef vtx = graph.addVertex(computeSet, "AppendState");
  graph.setCycleEstimate(vtx, 100);

  Tensor tmp = states.reshape({states.shape()[0], states.shape()[1] * states.shape()[2]});
  Tensor newStates = graph.addVariable(FLOAT, tmp.shape(), "newStates" + std::to_string(tile));

  graph.connect(vtx["state"], state.reshape({state.shape()[0] * state.shape()[1]}));
  graph.connect(vtx["states"], tmp);
  graph.connect(vtx["newStates"], newStates);
  graph.connect(vtx["itr"], itr);

  graph.setTileMapping(vtx, tile);
  graph.setTileMapping(tmp, tile);
  graph.setTileMapping(newStates, tile);

  return std::make_pair(computeSet, newStates.reshape({states.shape()[0], states.shape()[1], states.shape()[2]}));
}

std::tuple<ComputeSet, Tensor> KalmanFilter::inverse(Graph & graph, const Tensor & in, uint tile, uint dim)
{
  ComputeSet computeSet = graph.addComputeSet("inverse" + std::to_string(tile));
  VertexRef vtx = graph.addVertex(computeSet, "MatrixInverse" + std::to_string(dim));
  graph.setCycleEstimate(vtx, 100);

  Tensor out = graph.addVariable(FLOAT, {dim, dim}, "invOut" + std::to_string(tile));

  graph.connect(vtx["in"], in);
  graph.connect(vtx["out"], out);
  graph.setTileMapping(vtx, tile);
  graph.setTileMapping(out, tile);

  return std::make_pair(computeSet, out);
}

std::tuple<ComputeSet, Tensor> KalmanFilter::packHits(Graph & graph, const Tensor & inputHits, const Tensor & itr, uint tile)
{
  ComputeSet computeSet = graph.addComputeSet("packHits4" + std::to_string(tile));
  VertexRef vtx = graph.addVertex(computeSet, "PackHits4");
  graph.setCycleEstimate(vtx, 10);

  // Same shape as ps
  Tensor out = graph.addVariable(FLOAT, {4, 1}, "hitsThisItr" + std::to_string(tile));

  graph.connect(vtx["inputHits"], inputHits);
  graph.connect(vtx["itr"], itr);
  graph.connect(vtx["out"], out);
  graph.setTileMapping(vtx, tile);
  graph.setTileMapping(out, tile);

  return std::make_pair(computeSet, out);
}

std::tuple<ComputeSet, Tensor> KalmanFilter::iterate(Graph & graph, const Tensor & iterator, uint tile)
{
  ComputeSet computeSet = graph.addComputeSet("iterate" + std::to_string(tile));
  VertexRef vtx = graph.addVertex(computeSet, "Iterate");
  graph.setCycleEstimate(vtx, 10);

  Tensor out = graph.addVariable(INT, {1}, "itrOut" + std::to_string(tile));

  graph.connect(vtx["in"], iterator);
  graph.connect(vtx["out"], out);
  graph.setTileMapping(vtx, tile);
  graph.setTileMapping(out, tile);

  return std::make_pair(computeSet, out);
}

std::tuple<ComputeSet, Tensor> KalmanFilter::product(Graph & graph, const Tensor & first, const Tensor & second, uint tile)
{
  ComputeSet computeSet = graph.addComputeSet("product" + std::to_string(tile));
  VertexRef vtx = graph.addVertex(computeSet, "MatrixProduct");
  graph.setCycleEstimate(vtx, 100);

  uint outRows = first.shape()[0];
  uint outCols = second.shape()[1];

  Tensor out = graph.addVariable(FLOAT, {outRows, outCols}, "prodOut" + std::to_string(tile));

  graph.connect(vtx["first"], first);
  graph.connect(vtx["second"], second);
  graph.connect(vtx["out"], out);
  graph.setTileMapping(vtx, tile);
  graph.setTileMapping(out, tile);

  return std::make_pair(computeSet, out);
}

std::tuple<ComputeSet, Tensor> KalmanFilter::scaledAdd(Graph & graph, const Tensor & first, const Tensor & second, uint tile, float s, std::string name)
{
  ComputeSet computeSet = graph.addComputeSet("scaledAdd" + name +  std::to_string(tile));
  VertexRef vtx = graph.addVertex(computeSet, "ScaledAdd");
  graph.setCycleEstimate(vtx, 100);

  uint outRows = first.shape()[0];
  uint outCols = first.shape()[1];

  Tensor out = graph.addVariable(FLOAT, {outRows, outCols}, "addOut" + name + std::to_string(tile));

  graph.connect(vtx["first"], first);
  graph.connect(vtx["second"], second);
  graph.connect(vtx["s"], s);
  graph.connect(vtx["out"], out);
  graph.setTileMapping(vtx, tile);
  graph.setTileMapping(out, tile);

  return std::make_pair(computeSet, out);
}

std::tuple<Sequence, std::vector<Tensor>, std::vector<Tensor>> KalmanFilter::filter(Graph & graph,
  const std::vector<Tensor> & H, const std::vector<Tensor> & G, const std::vector<Tensor> & hits,
  const std::vector<Tensor> & p, const std::vector<Tensor> & C)
{
  Sequence op;

  std::vector<Tensor> outp(p.size());
  std::vector<Tensor> outc(p.size());

  for (uint i = 0; i < p.size(); i++) {

    outp[i] = graph.addVariable(FLOAT, {4, 1});
    graph.setTileMapping(outp[i], i);

    outc[i] = graph.addVariable(FLOAT, {4, 4});
    graph.setTileMapping(outc[i], i);

    auto [computeHG, HG] = product(graph, H[i], G[i], i);
    op.add(Execute(computeHG));

    // inv_C_proj + HG @ H
    auto [computeHGH, HGH] = product(graph, HG, H[i], i);
    op.add(Execute(computeHGH));

    auto [computeInverseC, invOutC] = inverse(graph, C[i], i, 4);
    op.add(Execute(computeInverseC));

    auto [computeAddCHGH, CHGH_HGH] = scaledAdd(graph, invOutC, HGH, i);
    op.add(Execute(computeAddCHGH));

    auto [computeInverseCHGH, invOutCHGH] = inverse(graph, CHGH_HGH, i, 4);
    op.add(Execute(computeInverseCHGH));

    auto [computeHGhits, HGhits] = product(graph, HG, hits[i], i);
    op.add(Execute(computeHGhits));

    auto [computePFilt, Cp_filt] = product(graph, invOutC, p[i], i);
    op.add(Execute(computePFilt));

    auto [computeAddCpHits, Cp_filt_hits] = scaledAdd(graph, Cp_filt, HGhits, i);
    op.add(Execute(computeAddCpHits));

    auto [computeCHGHPFilt, CHGHPFilt] = product(graph, invOutCHGH, Cp_filt_hits, i);
    op.add(Execute(computeCHGHPFilt));

    op.add(Copy(CHGHPFilt, outp[i]));
    op.add(Copy(invOutCHGH, outc[i]));
  }

  return std::make_tuple(op, outp, outc);
}

std::tuple<Sequence, std::vector<Tensor>> KalmanFilter::propagateState(Graph & graph, std::vector<Tensor> & ps, std::vector<Tensor> & d)
{

  Sequence op;

  std::vector<Tensor> outP(ps.size());

  for (uint i = 0; i < ps.size(); i++) {

    outP[i] = graph.addVariable(FLOAT, {2, 1});
    graph.setTileMapping(outP[i], i);

    // Need to allocate this earlier, and not during every function invocation!
    Tensor updMFlat = graph.addConstant<float>(FLOAT, {4, 1}, {0., 1., 0., 0.});
    Tensor updM = graph.addVariable(FLOAT, {2, 2});

    graph.setTileMapping(updMFlat, i);
    graph.setTileMapping(updM, i);

    updM = updMFlat.reshape({2, 2}); // Can't seem to set nD constant

    graph.setTileMapping(updM, i);

    // xTan = tan([x0, x1])
    Tensor pTan = popops::map(graph, popops::expr::UnaryOpType::TAN, ps[i], op);
    graph.setTileMapping(pTan, i);

    // x += d * [[0, 1], [0, 0]] * xTan -> x0 += d * tan(x1), x1 += 0

    auto [computeProdUpdTan, updTan] = product(graph, updM, pTan, i);
    op.add(Execute(computeProdUpdTan));

    auto [computeProdUpdD, updD] = product(graph, updTan, d[i], i);
    op.add(Execute(computeProdUpdD));

    auto [computeAddP, addP] = scaledAdd(graph, ps[i], updD, i);
    op.add(Execute(computeAddP));

    op.add(Copy(addP, outP[i]));

  }

  return std::make_tuple(op, outP);

}

std::tuple<Sequence, std::vector<Tensor>> KalmanFilter::jacobian(Graph & graph, const std::vector<Tensor> & ps, const std::vector<Tensor> & d)
{

  Sequence op;

  std::vector<Tensor> jacs(ps.size());

  for (uint i = 0; i < ps.size(); i++) {

    Tensor updMFlat = graph.addConstant<float>(FLOAT, {4, 1}, {0., 1., 0., 0.});
    Tensor updM = graph.addVariable(FLOAT, {2, 2});

    jacs[i] = graph.addVariable(FLOAT, {2, 2});

    graph.setTileMapping(updMFlat, i);
    graph.setTileMapping(updM, i);
    graph.setTileMapping(jacs[i], i);

    updM = updMFlat.reshape({2, 2});

    Tensor xCos = popops::map(graph, popops::expr::UnaryOpType::COS, ps[i], op);
    Tensor xCosSq = popops::map(graph, popops::expr::UnaryOpType::SQUARE, xCos, op);
    Tensor xCosSqI = popops::map(graph, popops::expr::UnaryOpType::INVERSE, xCosSq, op);

    Tensor xCosM = popops::pad(graph, xCosSqI,
                               1, // pad lower
                               0, // pad upper,
                               1, // pad dim
                               0  // pad value
                              );

    graph.setTileMapping(xCos, i);
    graph.setTileMapping(xCosM, i);
    graph.setTileMapping(xCosSq, i);
    graph.setTileMapping(xCosSqI, i);

    auto [computeJac, jac] = product(graph, updM, xCosM, i);
    op.add(Execute(computeJac));

    Tensor updM2Flat = graph.addConstant<float>(FLOAT, {4, 1}, {1., 0., 0., 1.});
    Tensor updM2 = graph.addVariable(FLOAT, {2, 2});
    graph.setTileMapping(updM2Flat, i);
    graph.setTileMapping(updM2, i);
    updM2 = updM2Flat.reshape({2, 2});

    auto [computeAddJacUpd, jac_upd] = scaledAdd(graph, jac, updM2, i);
    op.add(Execute(computeAddJacUpd));

    op.add(Copy(jac_upd, jacs[i]));
  }

  return std::make_tuple(op, jacs);

}

std::tuple<Sequence, std::vector<Tensor>> KalmanFilter::projectEKF(Graph & graph, const std::vector<Tensor> & jacs,
                                const std::vector<Tensor> & qs,
                                      std::vector<Tensor> & covs)
{
  Sequence op;

  std::vector<Tensor> outC(covs.size());

  for (uint i = 0; i < jacs.size(); i++) {

    outC[i] = graph.addVariable(FLOAT, {2, 2});
    graph.setTileMapping(outC[i], i);

    auto [computeCovT1, covT1] = product(graph, jacs[i], covs[i], i);
    op.add(Execute(computeCovT1));

    Tensor jacsT = jacs[i].transpose();
    graph.setTileMapping(jacsT, i);

    auto [computeCovT2, covT2] = product(graph, covT1, jacsT, i);
    op.add(Execute(computeCovT2));

    auto [computeAddCovQ, covQ] = scaledAdd(graph, covT2, qs[i], i);
    op.add(Execute(computeAddCovQ));

    op.add(Copy(covQ, outC[i]));

  }

  return std::make_tuple(op, outC);

}

std::tuple<Sequence, std::vector<Tensor>, std::vector<Tensor>> KalmanFilter::project(Graph & graph, const std::vector<Tensor> & ps,
  const std::vector<Tensor> & C, const std::vector<Tensor> & F, const std::vector<Tensor> & Q)
{
  Sequence op;

  std::vector<Tensor> outP(ps.size());
  std::vector<Tensor> outC(ps.size());

  for (uint i = 0; i < outP.size(); i++) {


    auto [computePproj, p_proj] = product(graph, F[i], ps[i], i);
    op.add(Execute(computePproj));

    auto [computeFC, fc] = product(graph, F[i], C[i], i);
    op.add(Execute(computeFC));

    // Check this works wrt tile mapping
    auto [computeFCF, fcf] = product(graph, fc, F[i].transpose(), i);
    op.add(Execute(computeFCF));

    auto [computeCproj, c_proj] = scaledAdd(graph, fcf, Q[i], i);
    op.add(Execute(computeCproj));

    outC[i] = graph.addVariable(FLOAT, {4, 4});
    outP[i] = graph.addVariable(FLOAT, {4, 1});

    graph.setTileMapping(outP[i], i);
    graph.setTileMapping(outC[i], i);

    op.add(Copy(c_proj, outC[i]));
    op.add(Copy(p_proj, outP[i]));

  }

  return std::make_tuple(op, outP, outC);

}

std::tuple<Sequence, std::vector<Tensor>> KalmanFilter::packIterationTensors(Graph & graph, const std::vector<Tensor> & loop,
                              const std::vector<Tensor> & scatterInto,
                              const std::vector<Tensor> & inputs)
{
  Sequence op;
  std::vector<Tensor> hits_packed(inputs.size());

  for (uint i = 0; i < inputs.size(); i++) {

    std::string iStr = std::to_string(i);

    hits_packed[i] = graph.addVariable(FLOAT, {4, 1}, "hits_packed" + iStr);
    graph.setTileMapping(hits_packed[i], i);

    auto [computeH, h] = packHits(graph, inputs[i], loop[i], i);
    op.add(Execute(computeH));
    op.add(Copy(h, hits_packed[i]));

  }

  return std::make_tuple(op, hits_packed);

}

// Residual at step t
std::tuple<Sequence, std::vector<Tensor>>
  KalmanFilter::calcResidual(Graph & graph, const std::vector<Tensor> & hits,
                              const std::vector<Tensor> & p_filt,
                              const std::vector<Tensor> & H)
{
  Sequence op;
  std::vector<Tensor> res_out(p_filt.size());

  for (uint i = 0; i < res_out.size(); i++) {

    std::string iStr = std::to_string(i);

    res_out[i] = graph.addVariable(FLOAT, {4, 1}, "res" + iStr);
    graph.setTileMapping(res_out[i], i);

    auto [computeHp, Hp] = product(graph, H[i], p_filt[i], i);
    op.add(Execute(computeHp));

    // Scaled add with prefactor of -1
    auto [computeRes, res] = scaledAdd(graph, hits[i], Hp, i, -1.0, "res");
    op.add(Execute(computeRes));
    op.add(Copy(res, res_out[i]));

  }

  return std::make_tuple(op, res_out);
}

// ChiSq for a plane of hits
std::tuple<Sequence, std::vector<Tensor>>
  KalmanFilter::calcChiSq(Graph & graph, const std::vector<Tensor> & res_filt,
                           const std::vector<Tensor> & G,
                           const std::vector<Tensor> & C_proj,
                           const std::vector<Tensor> & p_proj,
                           const std::vector<Tensor> & p_filt)
{
  Sequence op;
  std::vector<Tensor> chiSq(p_filt.size());

  for (uint i = 0; i < chiSq.size(); i++) {

    std::string iStr = std::to_string(i);

    Tensor rT = res_filt[i].transpose();
    graph.setTileMapping(rT, i);

    auto [computeRTG, rTG] = product(graph, rT, G[i], i);
    op.add(Execute(computeRTG));

    auto [computeResTerm, resTerm] = product(graph, rTG, res_filt[i], i);
    op.add(Execute(computeResTerm));

    auto [computeStateDiff, stateDiff] = scaledAdd(graph, p_filt[i], p_proj[i], i, -1.0);
    op.add(Execute(computeStateDiff));
    auto [computeInverseC, C_inv] = inverse(graph, C_proj[i], i, 4);
    op.add(Execute(computeInverseC));

    Tensor diffT = stateDiff.transpose();
    graph.setTileMapping(diffT, i);

    auto [computeStateTC, stateTC] = product(graph, diffT, C_inv, i);
    op.add(Execute(computeStateTC));

    auto [computeStateTerm, stateTerm] = product(graph, stateTC, stateDiff, i);
    op.add(Execute(computeStateTerm));

    auto [computeChiSqTot, chiSqTot] = scaledAdd(graph, resTerm, stateTerm, i, 1.0);
    op.add(Execute(computeChiSqTot));

    chiSq[i] = graph.addVariable(FLOAT, {1}, "chiSq" + iStr);
    graph.setTileMapping(chiSq[i], i);

    op.add(Copy(chiSqTot, chiSq[i]));

  }

  return std::make_tuple(op, chiSq);
}

std::tuple<Sequence, std::vector<Tensor>> KalmanFilter::chiSqTest(Graph & graph,
                                                    const std::vector<Tensor> & chiSq,
                                                    const std::vector<Tensor> & threshold)
{

  Sequence op;
  std::vector<Tensor> chiSqPred(chiSq.size());

  for (uint i = 0; i < chiSqPred.size(); i++) {

    std::string iStr = std::to_string(i);

    chiSqPred[i] = popops::map(graph, popops::expr::BinaryOpType::GREATER_THAN, chiSq[i], threshold[i], op);
    graph.setTileMapping(chiSqPred[i], i);
  }

  return std::make_tuple(op, chiSqPred);
}

std::tuple<Sequence, std::vector<Tensor>, std::vector<Tensor>>
  KalmanFilter::smooth(Graph & graph, const std::vector<Tensor> & p_smooth_prev,
                        const std::vector<Tensor> & C_smooth_prev,
                        const std::vector<Tensor> & p_filt,
                        const std::vector<Tensor> & C_filt,
                        const std::vector<Tensor> & p_proj,
                        const std::vector<Tensor> & C_proj,
                        const std::vector<Tensor> & F)
{
  Sequence op;

  std::vector<Tensor> p_smoothOut(p_filt.size());
  std::vector<Tensor> C_smoothOut(p_filt.size());

  for (uint i = 0; i < p_filt.size(); i++) {

    std::string iStr = std::to_string(i);

    // Compute A

    auto [computeInverseC_proj, invOutC_proj] = inverse(graph, C_proj[i], i, 4);
    op.add(Execute(computeInverseC_proj));

    auto [computeFC, fc] = product(graph, F[i].transpose(), invOutC_proj, i);
    op.add(Execute(computeFC));

    auto [computeA, a] = product(graph, C_filt[i], fc, i);
    op.add(Execute(computeA));


    auto [computePDiff, pDiff] = scaledAdd(graph, p_smooth_prev[i], p_proj[i], i, -1);
    op.add(Execute(computePDiff));

    auto [computeCDiff, cDiff] = scaledAdd(graph, C_smooth_prev[i], C_proj[i], i, -1);
    op.add(Execute(computeCDiff));

    auto [computeAPDiff, aPDiff] = product(graph, a, pDiff, i);
    op.add(Execute(computeAPDiff));

    auto [computeP_smooth, p_smooth] = scaledAdd(graph, p_filt[i], aPDiff, i);
    op.add(Execute(computeP_smooth));


    auto [computeCdiffA, cDiffA] = product(graph, cDiff, a.transpose(), i);
    op.add(Execute(computeCdiffA));

    auto [computeACdiffA, aCDiffA] = product(graph, a, cDiffA, i);
    op.add(Execute(computeACdiffA));

    auto [computeC_smooth, C_smooth] = scaledAdd(graph, C_filt[i], aCDiffA, i);
    op.add(Execute(computeC_smooth));

    p_smoothOut[i] = graph.addVariable(FLOAT, {4, 1}, "p_smooth" + iStr);
    C_smoothOut[i] = graph.addVariable(FLOAT, {4, 4}, "C_smooth" + iStr);

    graph.setTileMapping(p_smoothOut[i], i);
    graph.setTileMapping(C_smoothOut[i], i);

    op.add(Copy(p_smooth, p_smoothOut[i]));
    op.add(Copy(C_smooth, C_smoothOut[i]));

  }

  return std::make_tuple(op, p_smoothOut, C_smoothOut);
}
