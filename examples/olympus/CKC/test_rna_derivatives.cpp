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
      robotoc::CKCInfo("FrontLeft_ankle_inner", "FrontLeft_ankle_outer",
                       baumgarte_time_step),
      robotoc::CKCInfo("BackLeft_ankle_inner", "BackLeft_ankle_outer",
                       baumgarte_time_step),
      robotoc::CKCInfo("FrontRight_ankle_inner", "FrontRight_ankle_outer",
                       baumgarte_time_step),
      robotoc::CKCInfo("BackRight_ankle_inner", "BackRight_ankle_outer",
                       baumgarte_time_step)};

  robotoc::Robot robot(model_info);
  robot.setLowerJointPositionLimit(
      Eigen::VectorXd::Constant(robot.dimv() - 6, -3.14));
  robot.setUpperJointPositionLimit(
      Eigen::VectorXd::Constant(robot.dimv() - 6, 3.14));
  robot.setJointEffortLimit(Eigen::VectorXd::Constant(robot.dimv() - 6, 24.8));
  robot.setJointVelocityLimit(
      Eigen::VectorXd::Constant(robot.dimv() - 6, 31.0));
  robot.setGravity(-3.72);

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

  std::vector<robotoc::Robot::Vector2d> f_ckc;
  Eigen::VectorXd g(robot.dimf_ckc());
  int g_idx = 0;
  for (int i = 0; i < robot.numCKCs(); ++i) {

    f_ckc.push_back(robotoc::Robot::Vector2d::Random());

    g.segment<2>(g_idx) = f_ckc.back();
    g_idx += 2;
  }

  robot.setCKCForces(f_ckc);

  Eigen::VectorXd ID = Eigen::VectorXd::Zero(robot.dimv());
  Eigen::VectorXd C =
      Eigen::VectorXd::Zero(robot.dimf_ckc() + contact_status.dimf());
  Eigen::MatrixXd IDdq = Eigen::MatrixXd::Zero(robot.dimv(), robot.dimv());
  Eigen::MatrixXd IDdv = Eigen::MatrixXd::Zero(robot.dimv(), robot.dimv());
  Eigen::MatrixXd IDda = Eigen::MatrixXd::Zero(robot.dimv(), robot.dimv());
  Eigen::MatrixXd dCdq = Eigen::MatrixXd::Zero(
      robot.dimf_ckc() + contact_status.dimf(), robot.dimv());
  Eigen::MatrixXd dCdv = Eigen::MatrixXd::Zero(
      robot.dimf_ckc() + contact_status.dimf(), robot.dimv());
  Eigen::MatrixXd dCda = Eigen::MatrixXd::Zero(
      robot.dimf_ckc() + contact_status.dimf(), robot.dimv());

  Eigen::MatrixXd IDdq_fd = Eigen::MatrixXd::Zero(robot.dimv(), robot.dimv());
  Eigen::MatrixXd IDdv_fd = Eigen::MatrixXd::Zero(robot.dimv(), robot.dimv());
  Eigen::MatrixXd IDda_fd = Eigen::MatrixXd::Zero(robot.dimv(), robot.dimv());
  Eigen::MatrixXd IDdf_fd = Eigen::MatrixXd::Zero(
      robot.dimv(), robot.dimf_ckc() + contact_status.dimf());

  robot.RNEA(q, v, a, ID);
  robot.RNEADerivatives(q, v, a, IDdq, IDdv, IDda);
  robot.computeBaumgarteResidual(contact_status, C);
  robot.computeBaumgarteDerivatives(contact_status, dCdq, dCdv, dCda);

  float eps = 1.0e-8;

  int idx_f = 0;
  for (int i = 0; i < robot.numCKCs(); ++i) {
    for (int j = 0; j < 2; ++j) {
      std::vector<robotoc::Robot::Vector2d> f_ckc_petrubed = f_ckc;
      f_ckc_petrubed[i](j) += eps;
      robot.setContactForces(contact_status, f_contact);
      robot.setCKCForces(f_ckc_petrubed);
      Eigen::VectorXd ID_perturbed = Eigen::VectorXd::Zero(robot.dimv());
      robot.RNEA(q, v, a, ID_perturbed);
      IDdf_fd.col(idx_f) = (ID_perturbed - ID) / eps;
      idx_f++;
    }
  }

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 3; ++j) {
      std::vector<robotoc::Robot::Vector6d> f_contact_petrubed = f_contact;
      f_contact_petrubed[i](j) += eps;
      robot.setContactForces(contact_status, f_contact_petrubed);
      robot.setCKCForces(f_ckc);
      Eigen::VectorXd ID_perturbed = Eigen::VectorXd::Zero(robot.dimv());
      robot.RNEA(q, v, a, ID_perturbed);
      IDdf_fd.col(idx_f) = (ID_perturbed - ID) / eps;
      idx_f++;
    }
  }

  robot.setContactForces(contact_status, f_contact);
  robot.setCKCForces(f_ckc);

  for (int i = 0; i < robot.dimv(); ++i) {
    Eigen::VectorXd tau = Eigen::VectorXd::Zero(robot.dimv());
    tau(i) = 1;
    Eigen::VectorXd q_perturbed = Eigen::VectorXd::Zero(robot.dimq());
    Eigen::VectorXd ID_perturbed = Eigen::VectorXd::Zero(robot.dimv());
    robot.integrateConfiguration(q, tau, eps, q_perturbed);
    robot.updateKinematics(q_perturbed, v, a);
    robot.RNEA(q_perturbed, v, a, ID_perturbed);
    IDdq_fd.col(i) = (ID_perturbed - ID) / 1.0e-8;
  }

  for (int i = 0; i < robot.dimv(); ++i) {
    Eigen::VectorXd v_perturbed = v;
    v_perturbed(i) += eps;
    Eigen::VectorXd ID_perturbed = Eigen::VectorXd::Zero(robot.dimv());
    robot.RNEA(q, v_perturbed, a, ID_perturbed);
    IDdv_fd.col(i) = (ID_perturbed - ID) / 1.0e-8;
  }

  for (int i = 0; i < robot.dimv(); ++i) {
    Eigen::VectorXd a_perturbed = a;
    a_perturbed(i) += eps;
    Eigen::VectorXd ID_perturbed = Eigen::VectorXd::Zero(robot.dimv());
    robot.RNEA(q, v, a_perturbed, ID_perturbed);
    IDda_fd.col(i) = (ID_perturbed - ID) / 1.0e-8;
  }

  std::cout << "IDdq error: " << (IDdq - IDdq_fd).norm() << std::endl;
  std::cout << "IDdv error: " << (IDdv - IDdv_fd).norm() << std::endl;
  std::cout << "IDda error: " << (IDda - IDda_fd).norm() << std::endl;
  std::cout << "IDdf error: " << (dCda.transpose() + IDdf_fd).norm()
            << std::endl;
  return 0;
}