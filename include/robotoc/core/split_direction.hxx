#ifndef ROBOTOC_SPLIT_DIRECTION_HXX_
#define ROBOTOC_SPLIT_DIRECTION_HXX_

#include "robotoc/core/split_direction.hpp"

#include <cassert>

namespace robotoc {

inline void SplitDirection::setContactDimension(const int dimf) {
  assert(dimf >= 0);
  assert(dimf <= dxi_full_.size());
  dimf_ = dimf;
}

inline void SplitDirection::setImpact() {
  dimg_ = 0; // NO CKC constraint during impact phase ???
};

inline void SplitDirection::setSwitchingConstraintDimension(const int dims) {
  assert(dims >= 0);
  assert(dims <= dxi_full_.size());
  dims_ = dims;
}

inline Eigen::VectorBlock<Eigen::VectorXd> SplitDirection::dq() {
  assert(isDimensionConsistent());
  return dx.head(dimv_);
}

inline const Eigen::VectorBlock<const Eigen::VectorXd>
SplitDirection::dq() const {
  assert(isDimensionConsistent());
  return dx.head(dimv_);
}

inline Eigen::VectorBlock<Eigen::VectorXd> SplitDirection::dv() {
  assert(isDimensionConsistent());
  return dx.tail(dimv_);
}

inline const Eigen::VectorBlock<const Eigen::VectorXd>
SplitDirection::dv() const {
  assert(isDimensionConsistent());
  return dx.tail(dimv_);
}

inline Eigen::VectorBlock<Eigen::VectorXd> SplitDirection::dagf() {
  return dagf_full_.head(dimv_ + dimg_ + dimf_);
}

inline const Eigen::VectorBlock<const Eigen::VectorXd>
SplitDirection::dagf() const {
  return dagf_full_.head(dimv_ + dimg_ + dimf_);
}

inline Eigen::VectorBlock<Eigen::VectorXd> SplitDirection::da() {
  return dagf_full_.head(dimv_);
}

inline const Eigen::VectorBlock<const Eigen::VectorXd>
SplitDirection::da() const {
  return dagf_full_.head(dimv_);
}

inline Eigen::VectorBlock<Eigen::VectorXd> SplitDirection::ddvgf() {
  return dagf();
}

inline const Eigen::VectorBlock<const Eigen::VectorXd>
SplitDirection::ddvgf() const {
  return dagf();
}

inline Eigen::VectorBlock<Eigen::VectorXd> SplitDirection::ddv() {
  return da();
}

inline const Eigen::VectorBlock<const Eigen::VectorXd>
SplitDirection::ddv() const {
  return da();
}

inline Eigen::VectorBlock<Eigen::VectorXd> SplitDirection::dgf() {
  return dagf_full_.segment(dimv_, dimg_ + dimf_);
}

inline const Eigen::VectorBlock<const Eigen::VectorXd>
SplitDirection::dgf() const {
  return dagf_full_.segment(dimv_, dimg_ + dimf_);
}

inline Eigen::VectorBlock<Eigen::VectorXd> SplitDirection::df() {
  return dagf_full_.segment(dimv_ + dimg_, dimf_);
}

inline const Eigen::VectorBlock<const Eigen::VectorXd>
SplitDirection::df() const {
  return dagf_full_.segment(dimv_ + dimg_, dimf_);
}

inline Eigen::VectorBlock<Eigen::VectorXd> SplitDirection::dlmd() {
  assert(isDimensionConsistent());
  return dlmdgmm.head(dimv_);
}

inline const Eigen::VectorBlock<const Eigen::VectorXd>
SplitDirection::dlmd() const {
  assert(isDimensionConsistent());
  return dlmdgmm.head(dimv_);
}

inline Eigen::VectorBlock<Eigen::VectorXd> SplitDirection::dgmm() {
  assert(isDimensionConsistent());
  return dlmdgmm.tail(dimv_);
}

inline const Eigen::VectorBlock<const Eigen::VectorXd>
SplitDirection::dgmm() const {
  assert(isDimensionConsistent());
  return dlmdgmm.tail(dimv_);
}

inline Eigen::VectorBlock<Eigen::VectorXd> SplitDirection::dbetarhomu() {
  return dbetarhomu_full_.head(dimv_ + dimg_ + dimf_);
}

inline const Eigen::VectorBlock<const Eigen::VectorXd>
SplitDirection::dbetarhomu() const {
  return dbetarhomu_full_.head(dimv_ + dimg_ + dimf_);
}

inline Eigen::VectorBlock<Eigen::VectorXd> SplitDirection::dbeta() {
  return dbetarhomu_full_.head(dimv_);
}

inline const Eigen::VectorBlock<const Eigen::VectorXd>
SplitDirection::dbeta() const {
  return dbetarhomu_full_.head(dimv_);
}

inline Eigen::VectorBlock<Eigen::VectorXd> SplitDirection::drhomu() {
  return dbetarhomu_full_.segment(dimv_, dimg_ + dimf_);
}

inline const Eigen::VectorBlock<const Eigen::VectorXd>
SplitDirection::drhomu() const {
  return dbetarhomu_full_.segment(dimv_, dimg_ + dimf_);
}

inline Eigen::VectorBlock<Eigen::VectorXd> SplitDirection::dxi() {
  return dxi_full_.head(dims_);
}

inline const Eigen::VectorBlock<const Eigen::VectorXd>
SplitDirection::dxi() const {
  return dxi_full_.head(dims_);
}

inline void SplitDirection::setZero() {
  dx.setZero();
  du.setZero();
  dagf().setZero();
  dlmdgmm.setZero();
  dbetarhomu().setZero();
  dnu_passive.setZero();
  dxi().setZero();
  dts = 0.0;
  dts_next = 0.0;
}

inline int SplitDirection::dimf() const { return dimf_; }

inline int SplitDirection::dims() const { return dims_; }

inline int SplitDirection::dimg() const { return dimg_; }

} // namespace robotoc

#endif // ROBOTOC_SPLIT_OCP_DIRECTION_HXX_