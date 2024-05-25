#include "robotoc/robot/ckc.hpp"
#include "pinocchio/algorithm/frames-derivatives.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics-derivatives.hpp"

#include <cassert>

#include <stdexcept>

namespace robotoc {

CKC::CKC(const pinocchio::Model &model, const CKCInfo &info)
    : info_(info), dimv_(model.nv), jXf_0_(SE3::Identity()),
      jXf_1_(SE3::Identity()), J_frame_(Eigen::MatrixXd::Zero(6, model.nv)),
      vel_partial_dq_(Eigen::MatrixXd::Zero(3, model.nv)),
      frame_v_partial_dq_(Eigen::MatrixXd::Zero(6, model.nv)),
      frame_a_partial_dq_(Eigen::MatrixXd::Zero(6, model.nv)),
      frame_a_partial_dv_(Eigen::MatrixXd::Zero(6, model.nv)),
      frame_a_partial_da_(Eigen::MatrixXd::Zero(6, model.nv)) {
  if (!model.existFrame(info.frame_0)) {
    throw std::invalid_argument("[CKC] invalid argument: frame_0 '" +
                                info.frame_0 + "' does not exit!");
  }
  if (!model.existFrame(info.frame_0)) {
    throw std::invalid_argument("[CKC] invalid argument: frame '" +
                                info.frame_0 + "' does not exit!");
  }

  frame_0_idx_ = model.getFrameId(info.frame_0);
  frame_1_idx_ = model.getFrameId(info.frame_1);
  parent_0_joint_idx_ = model.frames[frame_0_idx_].parent;
  parent_1_joint_idx_ = model.frames[frame_1_idx_].parent;
  jXf_0_ = model.frames[frame_0_idx_].placement;
  jXf_1_ = model.frames[frame_1_idx_].placement;
  v_frame_.setZero();
  a_frame_.setZero();
  v_linear_skew_.setZero();
  v_angular_skew_.setZero();
  r_skew_.setZero();
  alpha_skew_.setZero();
}

void CKC::computeJointForceFromConstraintForce(
    const Eigen::Matrix3d &R_0, const Eigen::Matrix3d &R_1,
    const Eigen::Vector3d &constraint_force,
    pinocchio::container::aligned_vector<pinocchio::Force> &joint_forces)
    const {

  // transform force to the local frame and then move to joint frame
  int sgn = 1;
  joint_forces[parent_0_joint_idx_] = jXf_0_.act(pinocchio::Force(
      sgn * R_0.transpose() * constraint_force, Eigen::Vector3d::Zero()));
  sgn *= -1;
  joint_forces[parent_1_joint_idx_] = jXf_1_.act(pinocchio::Force(
      sgn * R_0.transpose() * constraint_force, Eigen::Vector3d::Zero()));
}

void CKC::setBaumgarteGains(const double baumgarte_position_gain,
                            const double baumgarte_velocity_gain) {
  if (baumgarte_position_gain < 0) {
    throw std::out_of_range("[CKC] invalid argument: 'baumgarte_position_gain' "
                            "must be non-negative!");
  }
  if (baumgarte_velocity_gain < 0) {
    throw std::out_of_range("[CKC] invalid argument: 'baumgarte_velocity_gain' "
                            "must be non-negative!");
  }
  info_.baumgarte_position_gain = baumgarte_position_gain;
  info_.baumgarte_velocity_gain = baumgarte_velocity_gain;
}

const CKCInfo &CKC::ckcInfo() const { return info_; }
const int &CKC::frame_0_idx() const { return frame_0_idx_; }
const int &CKC::frame_1_idx() const { return frame_1_idx_; }

} // namespace robotoc