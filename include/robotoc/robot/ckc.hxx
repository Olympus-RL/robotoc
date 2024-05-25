#ifndef ROBOTOC_CKC_HXX_
#define ROBOTOC_CKC_HXX_

#include "robotoc/robot/ckc.hpp"

#include "pinocchio/algorithm/frames-derivatives.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics-derivatives.hpp"

#include <cassert>

namespace robotoc {

inline void CKC::kuk() const {
  float a = 1.0;
  a = a + 1.0;
}

template <typename VectorType1>
inline void CKC::computeBaumgarteResidual(
    const pinocchio::Model &model, const pinocchio::Data &data,
    const Eigen::MatrixBase<VectorType1> &baumgarte_residual) const {
  assert(baumgarte_residual.size() == 3);

  const_cast<Eigen::MatrixBase<VectorType1> &>(baumgarte_residual).setZero();
  int sgn = 1;

  ///  for evaluating we can use local world aligned
  for (int frame_idx : {frame_0_idx_, frame_1_idx_}) {
    (const_cast<Eigen::MatrixBase<VectorType1> &>(baumgarte_residual))
        .noalias() +=
        sgn * pinocchio::getFrameClassicalAcceleration(
                  model, data, frame_idx, pinocchio::LOCAL_WORLD_ALIGNED)
                  .linear();
    (const_cast<Eigen::MatrixBase<VectorType1> &>(baumgarte_residual))
        .noalias() +=
        sgn * info_.baumgarte_velocity_gain *
        pinocchio::getFrameVelocity(model, data, frame_idx,
                                    pinocchio::LOCAL_WORLD_ALIGNED)
            .linear();
    (const_cast<Eigen::MatrixBase<VectorType1> &>(baumgarte_residual))
        .noalias() += sgn * info_.baumgarte_position_gain *
                      (data.oMf[frame_idx].translation());
    sgn *= -1;
  };
}

template <typename MatrixType1, typename MatrixType2, typename MatrixType3>
inline void CKC::computeBaumgarteDerivatives(
    const pinocchio::Model &model, pinocchio::Data &data,
    const Eigen::MatrixBase<MatrixType1> &baumgarte_partial_dq,
    const Eigen::MatrixBase<MatrixType2> &baumgarte_partial_dv,
    const Eigen::MatrixBase<MatrixType3> &baumgarte_partial_da) {
  assert(baumgarte_partial_dq.cols() == dimv_);
  assert(baumgarte_partial_dv.cols() == dimv_);
  assert(baumgarte_partial_da.cols() == dimv_);
  assert(baumgarte_partial_dq.rows() == 3);
  assert(baumgarte_partial_dv.rows() == 3);
  assert(baumgarte_partial_da.rows() == 3);

  const_cast<Eigen::MatrixBase<MatrixType1> &>(baumgarte_partial_dq).setZero();
  const_cast<Eigen::MatrixBase<MatrixType2> &>(baumgarte_partial_dv).setZero();
  const_cast<Eigen::MatrixBase<MatrixType3> &>(baumgarte_partial_da).setZero();

  int sgn = 1;
  for (int frame_idx : {frame_0_idx_, frame_1_idx_}) {

    // zero out the matrices
    J_frame_.setZero();
    frame_a_partial_da_.setZero();
    frame_a_partial_dq_.setZero();
    frame_a_partial_dv_.setZero();
    frame_v_partial_dq_.setZero();
    v_frame_.setZero();
    a_frame_.setZero();
    v_linear_skew_.setZero();
    v_angular_skew_.setZero();
    alpha_skew_.setZero();
    r_skew_.setZero();

    pinocchio::getFrameAccelerationDerivatives(
        model, data, frame_idx, pinocchio::WORLD, frame_v_partial_dq_,
        frame_a_partial_dq_, frame_a_partial_dv_, frame_a_partial_da_);
    // Skew matrices and LOCAL_WORLD_ALIGNED frame Jacobian are needed to
    // convert the frame acceleration derivatives into the "classical"
    // acceleration derivatives.

    pinocchio::getFrameJacobian(model, data, frame_idx,
                                pinocchio::LOCAL_WORLD_ALIGNED, J_frame_);
    v_frame_ = pinocchio::getFrameVelocity(model, data, frame_idx,
                                           pinocchio::LOCAL_WORLD_ALIGNED);
    a_frame_ = pinocchio::getFrameAcceleration(model, data, frame_idx,
                                               pinocchio::LOCAL_WORLD_ALIGNED);
    pinocchio::skew(v_frame_.linear(), v_linear_skew_);
    pinocchio::skew(v_frame_.angular(), v_angular_skew_);
    pinocchio::skew(data.oMf[frame_idx].translation(), r_skew_);
    pinocchio::skew(a_frame_.angular(), alpha_skew_);

    // derivative of instantainious vel, not spatial
    vel_partial_dq_ = (frame_v_partial_dq_.template topRows<3>() -
                       r_skew_ * frame_v_partial_dq_.template bottomRows<3>() +
                       v_angular_skew_ * J_frame_.template topRows<3>());

    const_cast<Eigen::MatrixBase<MatrixType1> &>(baumgarte_partial_dq) +=
        sgn * frame_a_partial_dq_.template topRows<3>(); // spatial
    const_cast<Eigen::MatrixBase<MatrixType1> &>(baumgarte_partial_dq)
        .noalias() += sgn * v_angular_skew_ * vel_partial_dq_;
    const_cast<Eigen::MatrixBase<MatrixType1> &>(baumgarte_partial_dq)
        .noalias() -=
        sgn * v_linear_skew_ * frame_v_partial_dq_.template bottomRows<3>();
    const_cast<Eigen::MatrixBase<MatrixType1> &>(baumgarte_partial_dq)
        .noalias() += sgn * alpha_skew_ * J_frame_.template topRows<3>();
    const_cast<Eigen::MatrixBase<MatrixType1> &>(baumgarte_partial_dq)
        .noalias() -=
        sgn * r_skew_ * frame_a_partial_dq_.template bottomRows<3>();

    const_cast<Eigen::MatrixBase<MatrixType2> &>(baumgarte_partial_dv)
        .noalias() += sgn * frame_a_partial_dv_.template topRows<3>();
    const_cast<Eigen::MatrixBase<MatrixType2> &>(baumgarte_partial_dv)
        .noalias() += sgn * v_angular_skew_ * J_frame_.template topRows<3>();
    const_cast<Eigen::MatrixBase<MatrixType2> &>(baumgarte_partial_dv)
        .noalias() -= sgn * v_linear_skew_ * J_frame_.template bottomRows<3>();
    const_cast<Eigen::MatrixBase<MatrixType2> &>(baumgarte_partial_dv)
        .noalias() -=
        sgn * r_skew_ * frame_a_partial_dv_.template bottomRows<3>();

    const_cast<Eigen::MatrixBase<MatrixType3> &>(baumgarte_partial_da)
        .noalias() +=
        sgn * (frame_a_partial_da_.template topRows<3>() -
               r_skew_ * frame_a_partial_da_.template bottomRows<3>());
    (const_cast<Eigen::MatrixBase<MatrixType1> &>(baumgarte_partial_dq))
        .noalias() += sgn * info_.baumgarte_velocity_gain * vel_partial_dq_;

    (const_cast<Eigen::MatrixBase<MatrixType2> &>(baumgarte_partial_dv))
        .noalias() +=
        sgn * info_.baumgarte_velocity_gain * (J_frame_.template topRows<3>());
    (const_cast<Eigen::MatrixBase<MatrixType1> &>(baumgarte_partial_dq))
        .noalias() +=
        sgn * info_.baumgarte_position_gain * J_frame_.template topRows<3>();

    sgn *= -1;
  }
}

template <typename VectorType1>
inline void CKC::computeCKCResidual(
    const pinocchio::Model &model, const pinocchio::Data &data,
    const Eigen::MatrixBase<VectorType1> &ckc_residual) const {
  assert(ckc_residual.size() == 3);
  const_cast<Eigen::MatrixBase<VectorType1> &>(ckc_residual).setZero();
  int sgn = 1;
  for (int frame_idx : {frame_0_idx_}) {
    const_cast<Eigen::MatrixBase<VectorType1> &>(ckc_residual).noalias() +=
        sgn * data.oMf[frame_idx].translation();
    sgn *= -1;
  }
}

} // namespace robotoc

#endif // ROBOTOC_CKC_HXX_