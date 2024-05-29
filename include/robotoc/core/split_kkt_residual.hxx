#ifndef ROBOTOC_SPLIT_KKT_RESIDUAL_HXX_
#define ROBOTOC_SPLIT_KKT_RESIDUAL_HXX_

#include "robotoc/core/split_kkt_residual.hpp"

#include <cmath>

namespace robotoc {

inline void SplitKKTResidual::setContactDimension(const int dimf_contact) {
  assert(dimf_contact >= 0);
  assert(dimf_contact <= P_full_.size());
  dimf_contact_ = dimf_contact;
  dimf_ = dimf_contact_ + dimf_ckc_;
}

inline void SplitKKTResidual::setCKCDimension(const int dimf_ckc) {
  assert(dimf_ckc >= 0);
  dimf_ckc_ = dimf_ckc;
  dimf_ = dimf_contact_ + dimf_ckc_;
}

inline void SplitKKTResidual::setSwitchingConstraintDimension(const int dims) {
  assert(dims >= 0);
  assert(dims <= P_full_.size());
  dims_ = dims;
}

inline Eigen::VectorBlock<Eigen::VectorXd> SplitKKTResidual::Fq() {
  return Fx.head(dimv_);
}

inline const Eigen::VectorBlock<const Eigen::VectorXd>
SplitKKTResidual::Fq() const {
  return Fx.head(dimv_);
}

inline Eigen::VectorBlock<Eigen::VectorXd> SplitKKTResidual::Fv() {
  return Fx.tail(dimv_);
}

inline const Eigen::VectorBlock<const Eigen::VectorXd>
SplitKKTResidual::Fv() const {
  return Fx.tail(dimv_);
}

inline Eigen::VectorBlock<Eigen::VectorXd> SplitKKTResidual::P() {
  return P_full_.head(dims_);
}

inline const Eigen::VectorBlock<const Eigen::VectorXd>
SplitKKTResidual::P() const {
  return P_full_.head(dims_);
}

inline Eigen::VectorBlock<Eigen::VectorXd> SplitKKTResidual::lq() {
  return lx.head(dimv_);
}

inline const Eigen::VectorBlock<const Eigen::VectorXd>
SplitKKTResidual::lq() const {
  return lx.head(dimv_);
}

inline Eigen::VectorBlock<Eigen::VectorXd> SplitKKTResidual::lv() {
  return lx.tail(dimv_);
}

inline const Eigen::VectorBlock<const Eigen::VectorXd>
SplitKKTResidual::lv() const {
  return lx.tail(dimv_);
}

inline Eigen::VectorBlock<Eigen::VectorXd> SplitKKTResidual::lf() {
  return lf_full_.head(dimf_);
}

inline const Eigen::VectorBlock<const Eigen::VectorXd>
SplitKKTResidual::lf() const {
  return lf_full_.head(dimf_);
}

inline Eigen::VectorBlock<Eigen::VectorXd> SplitKKTResidual::lf_contact() {
  return lf_full_.segment(dimf_ckc_, dimf_contact_);
}

inline const Eigen::VectorBlock<const Eigen::VectorXd>
SplitKKTResidual::lf_contact() const {
  return lf_full_.segment(dimf_ckc_, dimf_contact_);
}

inline double SplitKKTResidual::KKTError() const {
  double err = 0;
  err += Fx.squaredNorm();
  if (P().size() > 0) {
    err += P().squaredNorm();
  }
  err += lx.squaredNorm();
  err += lu.squaredNorm();
  err += la.squaredNorm();
  err += ldv.squaredNorm();
  if (lf().size() > 0) {
    err += lf().squaredNorm();
  }
  return err;
}

template <int p = 1> inline double SplitKKTResidual::primalFeasibility() const {
  double feasibility = Fx.template lpNorm<p>();
  if (dims_ > 0) {
    feasibility += P().template lpNorm<p>();
  }
  return feasibility;
}

template <int p = 1> inline double SplitKKTResidual::dualFeasibility() const {
  double feasibility = 0;
  feasibility += lx.template lpNorm<p>();
  feasibility += la.template lpNorm<p>();
  feasibility += ldv.template lpNorm<p>();
  if (lf().size() > 0)
    feasibility += lf().template lpNorm<p>();
  feasibility += lu.template lpNorm<p>();
  return feasibility;
}

inline void SplitKKTResidual::setZero() {
  Fx.setZero();
  if (P().size() > 0) {
    P().setZero();
  }
  lx.setZero();
  la.setZero();
  ldv.setZero();
  lu.setZero();
  if (lf().size() > 0) {
    lf().setZero();
  }
  h = 0.0;
}

inline int SplitKKTResidual::dimf() const { return dimf_; }

inline int SplitKKTResidual::dims() const { return dims_; }

} // namespace robotoc

#endif // ROBOTOC_SPLIT_KKT_RESIDUAL_HXX_