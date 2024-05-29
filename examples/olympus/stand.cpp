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

#include "robotoc/utils/ocp_benchmarker.hpp"
#include <vector>

int main(int argc, char *argv[]) {
  int i = 0;
  robotoc::RobotModelInfo model_info;
  model_info.urdf_path = "/home/bolivar/OLYMPOC/robotoc/descriptions/"
                         "olympus_description/urdf/olympus.urdf";
  model_info.base_joint_type = robotoc::BaseJointType::FloatingBase;
  const double baumgarte_time_step = 0.05;
  model_info.point_contacts = {
      robotoc::ContactModelInfo("FrontLeft_paw", baumgarte_time_step),
      robotoc::ContactModelInfo("BackLeft_paw", baumgarte_time_step),
      robotoc::ContactModelInfo("FrontRight_paw", baumgarte_time_step),
      robotoc::ContactModelInfo("BackRight_paw", baumgarte_time_step)};
  model_info.ckcs = {
      robotoc::CKCInfo("FrontLeft_ankle_inner", "FrontLeft_ankle_outer", 1, 1),
      robotoc::CKCInfo("BackLeft_ankle_inner", "BackLeft_ankle_outer", 1, 1),
      robotoc::CKCInfo("FrontRight_ankle_inner", "FrontRight_ankle_outer", 1,
                       1),
      robotoc::CKCInfo("BackRight_ankle_inner", "BackRight_ankle_outer", 1, 1)};

  model_info.contact_inv_damping = 1e-6;
  robotoc::Robot robot(model_info);
  robot.setLowerJointPositionLimit(
      Eigen::VectorXd::Constant(robot.dimv() - 6, -3.14));
  robot.setUpperJointPositionLimit(
      Eigen::VectorXd::Constant(robot.dimv() - 6, 3.14));
  robot.setJointEffortLimit(Eigen::VectorXd::Constant(robot.dimv() - 6, 24.8));
  robot.setJointVelocityLimit(
      Eigen::VectorXd::Constant(robot.dimv() - 6, 31.0));
  robot.setGravity(-0.0);

  const double dt = 0.02;
  const double standing_time = 0.2;
  const double t0 = 0;

  Eigen::VectorXd q = Eigen::VectorXd::Zero(robot.dimq());
  Eigen::VectorXd v = Eigen::VectorXd::Zero(robot.dimv());
  Eigen::VectorXd a = Eigen::VectorXd::Zero(robot.dimv());

  q = robot.generateFeasibleConfiguration();
  v.setRandom();
  a.setRandom();

  robot.updateKinematics(q, v, a);
  Eigen::VectorXd ckc_res = Eigen::VectorXd::Zero(robot.dimf_ckc());
  robot.computeCKCResidual(ckc_res);

  auto contact_status = robot.createContactStatus();
  for (int i = 0; i < robot.maxNumContacts(); ++i) {
    contact_status.activateContact(i);
  }

  Eigen::VectorXd f_stack = Eigen::VectorXd::Zero(robot.max_dimf());
  std::vector<robotoc::Robot::Vector6d> f_contact;
  int segement_start = 0;
  for (int i = 0; i < robot.maxNumContacts(); ++i) {
    f_contact.push_back(robotoc::Robot::Vector6d::Random());
    f_stack.segment<3>(segement_start) = f_contact.back().head(3);
    segement_start += 3;
  }
  robot.setContactForces(contact_status, f_contact);

  Eigen::VectorXd bamagrate_res = Eigen::VectorXd::Zero(robot.max_dimf());
  Eigen::MatrixXd dcdq = Eigen::MatrixXd::Zero(robot.max_dimf(), robot.dimv());
  Eigen::MatrixXd dcdv = Eigen::MatrixXd::Zero(robot.max_dimf(), robot.dimv());
  Eigen::MatrixXd dcda = Eigen::MatrixXd::Zero(robot.max_dimf(), robot.dimv());

  robot.computeBaumgarteResidual(contact_status, bamagrate_res);
  robot.computeBaumgarteDerivatives(contact_status, dcdq, dcdv, dcda);

  std::vector<robotoc::Robot::Vector2d> ckc_forces;
  Eigen::VectorXd g(robot.dimf_ckc());
  int g_idx = 0;
  for (int i = 0; i < robot.numCKCs(); ++i) {

    ckc_forces.push_back(robotoc::Robot::Vector2d::Random());

    g.segment<2>(g_idx) = ckc_forces.back();
    g_idx += 2;
  }

  robot.setCKCForces(ckc_forces);

  Eigen::MatrixXd Jckc = Eigen::MatrixXd::Zero(robot.dimf_ckc(), robot.dimv());
  Eigen::VectorXd tau(robot.dimv());
  tau.setZero();

  robot.RNEA(q, Eigen::VectorXd::Zero(robot.dimv()),
             Eigen::VectorXd::Zero(robot.dimv()), tau);
  robot.computeCKCJacobian(Jckc);

  Eigen::MatrixXd tau_ref = -Jckc.transpose() * g - dcda.transpose() * f_stack;
  std::cout << "tau: " << tau.transpose() << std::endl;
  std::cout << "tau_ref: " << tau_ref.transpose() << std::endl;
  std::cout << "tau - tau_ref norm: " << (tau - tau_ref).norm() << std::endl;

  Eigen::MatrixXd J0 = Eigen::MatrixXd::Zero(6, robot.dimv());
  Eigen::MatrixXd J1 = Eigen::MatrixXd::Zero(6, robot.dimv());

  return 0;
}