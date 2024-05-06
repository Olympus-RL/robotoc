#include <memory>
#include <string>

#include "Eigen/Core"

#include "robotoc/constraints/constraints.hpp"
#include "robotoc/constraints/friction_cone.hpp"
#include "robotoc/constraints/joint_position_lower_limit.hpp"
#include "robotoc/constraints/joint_position_upper_limit.hpp"
#include "robotoc/constraints/joint_torques_lower_limit.hpp"
#include "robotoc/constraints/joint_torques_upper_limit.hpp"
#include "robotoc/constraints/joint_velocity_lower_limit.hpp"
#include "robotoc/constraints/joint_velocity_upper_limit.hpp"
#include "robotoc/cost/com_cost.hpp"
#include "robotoc/cost/configuration_space_cost.hpp"
#include "robotoc/cost/cost_function.hpp"
#include "robotoc/cost/periodic_com_ref.hpp"
#include "robotoc/cost/periodic_swing_foot_ref.hpp"
#include "robotoc/cost/task_space_3d_cost.hpp"
#include "robotoc/ocp/ocp.hpp"
#include "robotoc/planner/contact_sequence.hpp"
#include "robotoc/robot/robot.hpp"
#include "robotoc/solver/ocp_solver.hpp"
#include "robotoc/solver/solver_options.hpp"
#include "robotoc/sto/sto_constraints.hpp"
#include "robotoc/sto/sto_cost_function.hpp"

/*
Eigen::VectorXd computeRtx() {
  Eigen::VectorXd Rtx = Eigen::VectorXd::Zero(contact_status.dimf());
  int i = 0;
  int dim = 0;
  for (robotoc::PointContact &contact : robot.robotModelInfo().point_contacts) {
    Eigen::Vector3d x = contact_status.contact_placements(i);
    Eigen::Matrix3d R = robot.getFrameRotation(contact.contact_frame_id());
    Rtx.segment(dim, 3) = R.transpose() * (x);
    i++;
    dim += 3;
  }
  return Rtx;
}; // Add a semicolon here

Eigen::MatrixXd computeRtxJacobian() {
  Eigen::MatrixXd dRtx_dq =
      Eigen::MatrixXd::Zero(contact_status.dimf(), robot.dimv());
  int i = 0;
  int dim = 0;
  for (robotoc::PointContact &contact : robot.robotModelInfo().point_contacts) {
    Eigen::Vector3d x = contact_status.contact_placements(i);
    Eigen::Matrix3d R = robot.getFrameRotation(contact.contact_frame_id());
    Eigen::Matrix3d Rtx_skew;
    Rtx_skew.setZero();
    Eigen::Vector3d rtx = R.transpose() * x;
    Rtx_skew << 0, -rtx(2), rtx(1), rtx(2), 0, -rtx(0), -rtx(1), rtx(0), 0;

    dRtx_dq.block(dim, 0, 3, robot.dimv()) =
        Rtx_skew * robot.getFrameJacobian(contact.contact_frame_id());
    i++;
    dim += 3;
  }
  return dRtx_dq;
}; // Add a semicolon here

*/

void test_finite_diff_contact(robotoc::Robot robot,
                              robotoc::ContactStatus contact_status,
                              const Eigen::VectorXd &q,
                              const Eigen::VectorXd &v, Eigen::VectorXd &a) {

  Eigen::VectorXd bg_res = Eigen::VectorXd::Zero(contact_status.dimf());

  Eigen::MatrixXd dCdq =
      Eigen::MatrixXd::Zero(contact_status.dimf(), robot.dimv());
  Eigen::MatrixXd dCdv =
      Eigen::MatrixXd::Zero(contact_status.dimf(), robot.dimv());
  Eigen::MatrixXd dCda =
      Eigen::MatrixXd::Zero(contact_status.dimf(), robot.dimv());
  Eigen::MatrixXd dRtx_dq =
      Eigen::MatrixXd::Zero(contact_status.dimf(), robot.dimv());

  Eigen::MatrixXd J = Eigen::MatrixXd::Zero(6, robot.dimv());

  Eigen::MatrixXd dCdq_fd =
      Eigen::MatrixXd::Zero(contact_status.dimf(), robot.dimv());
  Eigen::MatrixXd dCdv_fd =
      Eigen::MatrixXd::Zero(contact_status.dimf(), robot.dimv());
  Eigen::MatrixXd dCda_fd =
      Eigen::MatrixXd::Zero(contact_status.dimf(), robot.dimv());
  Eigen::MatrixXd dRtx_dq_fd =
      Eigen::MatrixXd::Zero(contact_status.dimf(), robot.dimv());

  Eigen::MatrixXd J_fd = Eigen::MatrixXd::Zero(6, robot.dimv());

  robot.updateKinematics(q, v, a);
  robot.computeBaumgarteResidual(contact_status, bg_res);
  robot.computeBaumgarteDerivatives(contact_status, dCdq, dCdv, dCda);
  robot.getFrameJacobian(robot.frameId("LF_FOOT"), J);

  const double eps = 1.0e-8;

  for (int i = 0; i < robot.dimv(); ++i) {
    Eigen::VectorXd d_tau =
        Eigen::VectorXd::Zero(robot.dimv()); // tangen space mtf
    d_tau(i) = 1;
    Eigen::VectorXd q_p = Eigen::VectorXd::Zero(robot.dimq());
    robot.integrateConfiguration(q, d_tau, eps, q_p);
    robot.updateKinematics(q_p, v, a);
    Eigen::VectorXd bg_res_p = Eigen::VectorXd::Zero(contact_status.dimf());
    robot.computeBaumgarteResidual(contact_status, bg_res_p);
    dCdq_fd.col(i) = (bg_res_p - bg_res) / eps;
  }

  for (int i = 0; i < robot.dimv(); ++i) {
    Eigen::VectorXd v_p = v;
    v_p(i) += eps;
    robot.updateKinematics(q, v_p, a);
    Eigen::VectorXd bg_res_p = Eigen::VectorXd::Zero(contact_status.dimf());
    robot.computeBaumgarteResidual(contact_status, bg_res_p);
    dCdv_fd.col(i) = (bg_res_p - bg_res) / eps;
  }

  for (int i = 0; i < robot.dimv(); ++i) {
    Eigen::VectorXd a_p = a;
    a_p(i) += eps;
    robot.updateKinematics(q, v, a_p);
    Eigen::VectorXd bg_res_p = Eigen::VectorXd::Zero(contact_status.dimf());
    robot.computeBaumgarteResidual(contact_status, bg_res_p);
    dCda_fd.col(i) = (bg_res_p - bg_res) / eps;
  }

  std::cout << "dCdq error: " << (dCdq - dCdq_fd).norm() << std::endl;
  std::cout << "dCdv error: " << (dCdv - dCdv_fd).norm() << std::endl;
  std::cout << "dCda error: " << (dCda - dCda_fd).norm() << std::endl;
}

int main(int argc, char *argv[]) {
  robotoc::RobotModelInfo model_info;
  model_info.urdf_path = "../anymal_b_simple_description/urdf/anymal.urdf";
  model_info.base_joint_type = robotoc::BaseJointType::FloatingBase;
  const double baumgarte_time_step = 0.05;
  const double baumgarte_velocity_gain = 1.0;
  const double baumgarte_position_gain = 1.0;

  model_info.point_contacts = {
      robotoc::ContactModelInfo("LF_FOOT", baumgarte_position_gain,
                                baumgarte_velocity_gain),
      robotoc::ContactModelInfo("LH_FOOT", baumgarte_position_gain,
                                baumgarte_velocity_gain),
      robotoc::ContactModelInfo("RF_FOOT", baumgarte_position_gain,
                                baumgarte_velocity_gain),
      robotoc::ContactModelInfo("RH_FOOT", baumgarte_position_gain,
                                baumgarte_velocity_gain),
  };
  robotoc::Robot robot(model_info);

  const double dt = 0.02;
  const Eigen::Vector3d jump_length = {0.8, 0, 0};
  const double flying_up_time = 0.15;
  const double flying_down_time = flying_up_time;
  const double flying_time = flying_up_time + flying_down_time;
  const double ground_time = 0.70;
  const double t0 = 0;

  Eigen::VectorXd q_standing(Eigen::VectorXd::Zero(robot.dimq()));
  q_standing << 0, 0, 0.4792, 0, 0, 0, 1, -0.1, 0.7, -1.0, -0.1, -0.7, 1.0, 0.1,
      0.7, -1.0, 0.1, -0.7, 1.0;

  // Create the contact sequence
  auto contact_sequence = std::make_shared<robotoc::ContactSequence>(robot);
  const double mu = 0.7;
  const std::unordered_map<std::string, double> friction_coefficients = {
      {"LF_FOOT", mu}, {"LH_FOOT", mu}, {"RF_FOOT", mu}, {"RH_FOOT", mu}};

  robot.updateFrameKinematics(q_standing);
  const Eigen::Vector3d x3d0_LF = robot.framePosition("LF_FOOT");
  const Eigen::Vector3d x3d0_LH = robot.framePosition("LH_FOOT");
  const Eigen::Vector3d x3d0_RF = robot.framePosition("RF_FOOT");
  const Eigen::Vector3d x3d0_RH = robot.framePosition("RH_FOOT");

  std::unordered_map<std::string, Eigen::Vector3d> contact_positions = {
      {"LF_FOOT", x3d0_LF},
      {"LH_FOOT", x3d0_LH},
      {"RF_FOOT", x3d0_RF},
      {"RH_FOOT", x3d0_RH}};
  auto contact_status_standing = robot.createContactStatus();
  contact_status_standing.activateContacts(
      std::vector<std::string>({"LF_FOOT", "LH_FOOT", "RF_FOOT", "RH_FOOT"}));
  contact_status_standing.setContactPlacements(contact_positions);
  contact_status_standing.setFrictionCoefficients(friction_coefficients);
  contact_sequence->init(contact_status_standing);

  auto contact_status_flying = robot.createContactStatus();
  contact_sequence->push_back(contact_status_flying, t0 + ground_time - 0.3,
                              true);

  contact_positions["LF_FOOT"].noalias() += jump_length;
  contact_positions["LH_FOOT"].noalias() += jump_length;
  contact_positions["RF_FOOT"].noalias() += jump_length;
  contact_positions["RH_FOOT"].noalias() += jump_length;
  contact_status_standing.setContactPlacements(contact_positions);
  contact_sequence->push_back(contact_status_standing,
                              t0 + ground_time + flying_time - 0.1, true);

  Eigen::VectorXd q = q_standing;
  Eigen::VectorXd v = Eigen::VectorXd::Zero(robot.dimv());
  Eigen::VectorXd a = Eigen::VectorXd::Zero(robot.dimv());

  std::cout
      << "Testing finite difference of contact constraints with robot stading"
      << std::endl;
  test_finite_diff_contact(robot, contact_status_standing, q, v, a);

  std::cout << "Testing finite difference of contact constraints  random config"
            << std::endl;

  for (int i = 0; i < 1; i++) {
    q = robot.generateFeasibleConfiguration(); // remember joint limits
    v = Eigen::VectorXd::Random(robot.dimv());
    a = Eigen::VectorXd::Random(robot.dimv());
    test_finite_diff_contact(robot, contact_status_standing, q, v, a);
  }
}