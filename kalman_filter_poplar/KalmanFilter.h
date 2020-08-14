// export POPLAR_ENGINE_OPTIONS='{"target.workerStackSizeInBytes":1024}'

#pragma once

#include <vector>

#include <poplar/DeviceManager.hpp>
#include <poplar/IPUModel.hpp>

using namespace poplar;
using namespace poplar::program;
using namespace popops;

namespace KalmanFilter {

Device connectToIPU();

std::tuple<ComputeSet, Tensor, Tensor, Tensor> skipSwitch(Graph & graph,
                                                  const Tensor & inputPOld,
                                                  const Tensor & inputCOld,
                                                  const Tensor & inputPNew,
                                                  const Tensor & inputCNew,
                                                  const Tensor & chiSq,
                                                  uint tile);

std::tuple<ComputeSet, Tensor> smoothingState(Graph & graph, const Tensor & states, const Tensor & itr, uint tile);

std::tuple<ComputeSet, Tensor> appendTo(Graph & graph, const Tensor & state, const Tensor & itr, Tensor & states, uint tile);

std::tuple<ComputeSet, Tensor> inverse(Graph & graph, const Tensor & in, uint tile, uint dim);

std::tuple<ComputeSet, Tensor> packHits(Graph & graph, const Tensor & inputHits, const Tensor & itr, uint tile);

std::tuple<ComputeSet, Tensor> iterate(Graph & graph, const Tensor & iterator, uint tile);
std::tuple<ComputeSet, Tensor> product(Graph & graph, const Tensor & first, const Tensor & second, uint tile);

std::tuple<ComputeSet, Tensor> scaledAdd(Graph & graph, const Tensor & first, const Tensor & second, uint tile, float s = 1.0, std::string name = "");

std::tuple<Sequence, std::vector<Tensor>, std::vector<Tensor>> filter(Graph & graph,
  const std::vector<Tensor> & H, const std::vector<Tensor> & G, const std::vector<Tensor> & hits,
  const std::vector<Tensor> & p, const std::vector<Tensor> & C);

std::tuple<Sequence, std::vector<Tensor>> propagateState(Graph & graph, std::vector<Tensor> & ps, std::vector<Tensor> & d);

std::tuple<Sequence, std::vector<Tensor>> jacobian(Graph & graph, const std::vector<Tensor> & ps, const std::vector<Tensor> & d);

std::tuple<Sequence, std::vector<Tensor>> projectEKF(Graph & graph, const std::vector<Tensor> & jacs,
                                const std::vector<Tensor> & qs,
                                      std::vector<Tensor> & covs);

std::tuple<Sequence, std::vector<Tensor>, std::vector<Tensor>> project(Graph & graph, const std::vector<Tensor> & ps,
  const std::vector<Tensor> & C, const std::vector<Tensor> & F, const std::vector<Tensor> & Q);

std::tuple<Sequence, std::vector<Tensor>> packIterationTensors(Graph & graph, const std::vector<Tensor> & loop,
                              const std::vector<Tensor> & scatterInto,
                              const std::vector<Tensor> & inputs);

// Residual at step t
std::tuple<Sequence, std::vector<Tensor>>
  calcResidual(Graph & graph, const std::vector<Tensor> & hits,
                              const std::vector<Tensor> & p_filt,
                              const std::vector<Tensor> & H);

std::tuple<Sequence, std::vector<Tensor>>
  calcChiSq(Graph & graph, const std::vector<Tensor> & res_filt,
                           const std::vector<Tensor> & G,
                           const std::vector<Tensor> & C_proj,
                           const std::vector<Tensor> & p_proj,
                           const std::vector<Tensor> & p_filt);

std::tuple<Sequence, std::vector<Tensor>> chiSqTest(Graph & graph,
                                                    const std::vector<Tensor> & chiSq,
                                                    const std::vector<Tensor> & threshold);

std::tuple<Sequence, std::vector<Tensor>, std::vector<Tensor>>
  smooth(Graph & graph, const std::vector<Tensor> & p_smooth_prev,
                        const std::vector<Tensor> & C_smooth_prev,
                        const std::vector<Tensor> & p_filt,
                        const std::vector<Tensor> & C_filt,
                        const std::vector<Tensor> & p_proj,
                        const std::vector<Tensor> & C_proj,
                        const std::vector<Tensor> & F);

} // namespace KalmanFilter
