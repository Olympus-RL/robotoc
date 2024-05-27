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
    : info_(info), dimq_(model.nq), dimv_(model.nv) {
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