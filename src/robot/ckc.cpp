#include "robotoc/robot/ckc.hpp"
#include "pinocchio/algorithm/frames-derivatives.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics-derivatives.hpp"
#include "pinocchio/algorithm/model.hpp"

#include <cassert>
#include <math.h>
#include <stdexcept>
#include <vector>
namespace robotoc {

CKC::CKC(const ::pinocchio::Model &model, const CKCInfo &info)
    : info_(info), dimq_(model.nq), dimv_(model.nv), jXf_0_(SE3::Identity()),
      jXf_1_(SE3::Identity()) {
  if (!model.existFrame(info.frame_0)) {
    throw ::std::invalid_argument("[CKC] invalid argument: frame_0 '" +
                                  info.frame_0 + "' does not exit!");
  }
  if (!model.existFrame(info.frame_1)) {
    throw ::std::invalid_argument("[CKC] invalid argument: frame '" +
                                  info.frame_1 + "' does not exit!");
  }

  frame_0_idx_ = model.getFrameId(info.frame_0);
  frame_1_idx_ = model.getFrameId(info.frame_1);
  joint_0_idx_ = model.frames[frame_0_idx_].parent;
  joint_1_idx_ = model.frames[frame_1_idx_].parent;
  if (frame_0_idx_ == frame_1_idx_) {
    throw ::std::invalid_argument(
        "[CKC] invalid argument: frame_0 and frame_1 must be different!");
  }
  ::std::vector<pinocchio::JointIndex> joints2keepID;
  joints2keepID.push_back(model.frames[frame_0_idx_].parent);
  joints2keepID.push_back(model.frames[frame_0_idx_].parent - 1);
  joints2keepID.push_back(model.frames[frame_1_idx_].parent);
  joints2keepID.push_back(model.frames[frame_1_idx_].parent - 1);

  ::std::vector<pinocchio::JointIndex> joints2lockID;
  for (int i = 1; i < model.njoints; i++) {
    if (::std::find(joints2keepID.begin(), joints2keepID.end(), i) ==
        joints2keepID.end()) {
      joints2lockID.push_back(i);
    }
  }
  int start_joint_idx =
      *::std::min_element(joints2keepID.begin(), joints2keepID.end());
  start_q_idx_ = model.joints[start_joint_idx].idx_q();
  start_v_idx_ = model.joints[start_joint_idx].idx_v();
  common_ancestor_joint_idx_ = start_joint_idx - 1;
  std::string common_ancetsor_frame_name;
  for (auto &frame : model.frames) {
    if (frame.parent == common_ancestor_joint_idx_) {
      common_ancetsor_frame_name = frame.name;
      break;
    }
  }

  Eigen::VectorXd q = ::pinocchio::neutral(model);
  q.segment(3, 4) << std::sqrt(2.0) / 2, 0.0, 0.0,
      std::sqrt(2.0) /
          2; // 90 degree rotation to make the constraint in the x-y plane
  ::pinocchio::buildReducedModel(model, joints2lockID, q, submodel_);
  subdata_ = pinocchio::Data(submodel_);

  frame_0_idx_ =
      submodel_.getFrameId(info.frame_0); // update the frae idx with the sub
                                          // model since they might have changed
  frame_1_idx_ = submodel_.getFrameId(info.frame_1);
  jXf_0_ = submodel_.frames[frame_0_idx_].placement;
  jXf_1_ = submodel_.frames[frame_1_idx_].placement;
  common_ancestor_frame_idx_ = submodel_.getFrameId(common_ancetsor_frame_name);

  v_frame_.setZero();
  a_frame_.setZero();
  v_linear_skew_.setZero();
  v_angular_skew_.setZero();
  alpha_skew_.setZero();
  r_skew_.setZero();
  vel_partial_dq_.setZero();
  frame_v_partial_dq_.setZero();
  frame_a_partial_dq_.setZero();
  frame_a_partial_dv_.setZero();
  frame_a_partial_da_.setZero();
  J_frame_.setZero();
  Jc_.setZero();
}

void CKC::computeJointForceFromConstraintForce(
    const Eigen::Vector2d &g,
    pinocchio::container::aligned_vector<pinocchio::Force> &joint_forces) {
  Eigen::Vector3d g_3d_world = Eigen::Vector3d::Zero();
  g_3d_world.head<2>() = g;
  Eigen::Vector3d g_3d_local =
      subdata_.oMf[frame_0_idx_].rotation().transpose() * g_3d_world;

  joint_forces[joint_0_idx_] +=
      jXf_0_.act(pinocchio::Force(g_3d_local, Eigen::Vector3d::Zero()));

  g_3d_local = subdata_.oMf[frame_1_idx_].rotation().transpose() * g_3d_world;

  joint_forces[joint_1_idx_] += jXf_1_.act(pinocchio::Force(
      -g_3d_local,
      Eigen::Vector3d::Zero())); // minus sign on force 2 ///note the force

  //// correction term

  joint_forces[common_ancestor_joint_idx_] += pinocchio::Force(
      Eigen::Vector3d::Zero(),
      subdata_.oMf[common_ancestor_frame_idx_].rotation().transpose() *
          ((subdata_.oMf[frame_1_idx_].translation() -
            subdata_.oMf[frame_0_idx_].translation())
               .cross(g_3d_world))); // zero out the force
}

void CKC::computeGeneralizedForceFromConstraintForce(const Eigen::Vector2d &g,
                                                     Eigen::VectorXd &Q) {
  assert(Q.size() == dimv_);

  // Note to be honest I dont know when and when not to use .noalias()
  pinocchio::getFrameJacobian(submodel_, subdata_, frame_0_idx_,
                              pinocchio::LOCAL_WORLD_ALIGNED, J_frame_);
  Jc_.noalias() = J_frame_.template topRows<2>();
  pinocchio::getFrameJacobian(submodel_, subdata_, frame_1_idx_,
                              pinocchio::LOCAL_WORLD_ALIGNED, J_frame_);
  Jc_.noalias() -= J_frame_.template topRows<2>();
  Q.segment<4>(start_v_idx_).noalias() = Jc_.transpose() * g;
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

const CKCInfo &CKC::ckcInfo() const { return info_; };

} // namespace robotoc