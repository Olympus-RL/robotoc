#include <memory>
#include <random>
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
#include "robotoc/dynamics/contact_dynamics.hpp"
#include "robotoc/ocp/ocp.hpp"
#include "robotoc/planner/contact_sequence.hpp"
#include "robotoc/robot/robot.hpp"
#include "robotoc/solver/ocp_solver.hpp"
#include "robotoc/solver/solver_options.hpp"
#include "robotoc/sto/sto_constraints.hpp"
#include "robotoc/sto/sto_cost_function.hpp"
#include "robotoc/utils/derivative_checker.hpp"

void test_contact_dyn(robotoc::Robot &robot,
                      robotoc::ContactStatus &contact_status) {
  robotoc::SplitSolution sol = robotoc::SplitSolution::Random(robot);
  robotoc::ContactDynamicsData data(robot);
  robotoc::SplitKKTResidual kkt_residual(robot);
  robotoc::ContactDynamicsData data_ref = data;

  robot.updateKinematics(sol.q, sol.v, sol.a);
  robotoc::evalContactDynamics(robot, contact_status, sol, data);
  robotoc::linearizeContactDynamics(robot, contact_status, sol, data_ref,
                                    kkt_residual);

  const double eps = 1.0e-8;

  Eigen::MatrixXd dCdq_ref =
      Eigen::MatrixXd::Zero(contact_status.dimf(), robot.dimv());

  for (int i = 0; i < robot.dimv(); ++i) {
    Eigen::VectorXd v_eps = Eigen::VectorXd::Zero(robot.dimv());
    v_eps(i) = 1;
    robotoc::SplitSolution sol_perturbed = sol;
    robot.integrateConfiguration(sol.q, v_eps, eps, sol_perturbed.q);
    robot.updateKinematics(sol_perturbed.q, sol_perturbed.v, sol_perturbed.a);
    robotoc::evalContactDynamics(robot, contact_status, sol_perturbed,
                                 data_ref);
    dCdq_ref.col(i) = (data_ref.C() - data.C()) / eps;
  }

  Eigen::MatrixXd dCdv_ref =
      Eigen::MatrixXd::Zero(contact_status.dimf(), robot.dimv());

  for (int i = 0; i < robot.dimv(); ++i) {
    robotoc::SplitSolution sol_perturbed = sol;
    sol_perturbed.v(i) += eps;
    robot.updateKinematics(sol_perturbed.q, sol_perturbed.v, sol_perturbed.a);
    robotoc::evalContactDynamics(robot, contact_status, sol_perturbed,
                                 data_ref);
    dCdv_ref.col(i) = (data_ref.C() - data.C()) / eps;
  }

  Eigen::MatrixXd dCda_ref =
      Eigen::MatrixXd::Zero(contact_status.dimf(), robot.dimv());

  for (int i = 0; i < robot.dimv(); ++i) {
    robotoc::SplitSolution sol_perturbed = sol;
    sol_perturbed.a(i) += eps;
    robot.updateKinematics(sol_perturbed.q, sol_perturbed.v, sol_perturbed.a);
    robotoc::evalContactDynamics(robot, contact_status, sol_perturbed,
                                 data_ref);
    dCda_ref.col(i) = (data_ref.C() - data.C()) / eps;
  }

  double fd_error_q = (dCdq_ref - data_ref.dCdq()).norm();
  double fd_error_v = (dCdv_ref - data_ref.dCdv()).norm();
  double fd_error_a = (dCda_ref - data_ref.dCda()).norm();
  std::cout << "fd_error_q: " << fd_error_q << std::endl;
  std::cout << "fd_error_v: " << fd_error_v << std::endl;
}

void test_config_cost(
    robotoc::Robot &robot,
    std::shared_ptr<robotoc::ConfigurationSpaceCost> config_cost) {
  // seed the random number generator
  srand((unsigned int)time(0));
  robotoc::SplitSolution sol = robotoc::SplitSolution::Random(robot);

  robotoc::DerivativeChecker derivative_checker(robot, 1e-8);
  std::cout << "stage cost" << std::endl;
  derivative_checker.checkFirstOrderStageCostDerivatives(config_cost);
  std::cout << "terminal cost" << std::endl;
  derivative_checker.checkFirstOrderImpactCostDerivatives(config_cost);
  std::cout << "impact cost" << std::endl;
  derivative_checker.checkFirstOrderTerminalCostDerivatives(config_cost);
}

int main(int argc, char *argv[]) {
  robotoc::RobotModelInfo model_info;
  const std::string urdf_file("/home/bolivar/OLYMPOC/robotoc/descriptions/"
                              "olympus_description/urdf/olympus_open.urdf");

  model_info.urdf_path = urdf_file;
  model_info.base_joint_type = robotoc::BaseJointType::FloatingBase;
  const double baumgarte_time_step = 0.05;
  model_info.point_contacts = {
      robotoc::ContactModelInfo("FrontLeft_paw", baumgarte_time_step),
      robotoc::ContactModelInfo("BackLeft_paw", baumgarte_time_step),
      robotoc::ContactModelInfo("FrontRight_paw", baumgarte_time_step),
      robotoc::ContactModelInfo("BackRight_paw", baumgarte_time_step)};

  robotoc::Robot robot(model_info);

  const double dt = 0.02;
  const Eigen::Vector3d jump_length = {0.8, 0, 0};
  const double flying_up_time = 0.15;
  const double flying_down_time = flying_up_time;
  const double flying_time = flying_up_time + flying_down_time;
  const double ground_time = 0.70;
  const double t0 = 0;

  // Create the cost function
  auto cost = std::make_shared<robotoc::CostFunction>();
  Eigen::VectorXd q_standing(Eigen::VectorXd::Zero(robot.dimq()));
  q_standing << 0, 0, 0.4792, 0, 0, 0, 1, -0.1, 0.7, -1.0, -0.1, -0.7, 1.0, 0.1,
      0.7, -1.0, 0.1, -0.7, 1.0;
  Eigen::VectorXd q_ref = q_standing;
  q_ref.head(3).noalias() += jump_length;
  Eigen::VectorXd q_weight(Eigen::VectorXd::Zero(robot.dimv()));
  q_weight << 1.0, 0, 0, 1.0, 1.0, 1.0, 0.001, 0.001, 0.001, 0.001, 0.001,
      0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001;
  Eigen::VectorXd v_weight = Eigen::VectorXd::Constant(robot.dimv(), 1.0);
  Eigen::VectorXd a_weight = Eigen::VectorXd::Constant(robot.dimv(), 1.0e-06);
  Eigen::VectorXd q_weight_impact(Eigen::VectorXd::Zero(robot.dimv()));
  q_weight_impact << 0, 0, 0, 100.0, 100.0, 100.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
      0.1, 0.1, 0.1, 0.1, 0.1, 0.1;
  Eigen::VectorXd v_weight_impact =
      Eigen::VectorXd::Constant(robot.dimv(), 1.0);
  Eigen::VectorXd dv_weight_impact =
      Eigen::VectorXd::Constant(robot.dimv(), 1.0e-06);
  auto config_cost = std::make_shared<robotoc::ConfigurationSpaceCost>(robot);
  config_cost->set_q_ref(q_ref);
  config_cost->set_q_weight(q_weight);
  config_cost->set_q_weight_terminal(q_weight);
  config_cost->set_q_weight_impact(q_weight_impact);
  config_cost->set_v_weight(v_weight);
  config_cost->set_v_weight_terminal(v_weight);
  config_cost->set_v_weight_impact(v_weight_impact);
  config_cost->set_dv_weight_impact(dv_weight_impact);
  config_cost->set_a_weight(a_weight);
  cost->add("config_cost", config_cost);

  // Create the constraints
  const double barrier_param = 1.0e-03;
  const double fraction_to_boundary_rule = 0.995;
  auto constraints = std::make_shared<robotoc::Constraints>(
      barrier_param, fraction_to_boundary_rule);
  auto joint_position_lower =
      std::make_shared<robotoc::JointPositionLowerLimit>(robot);
  auto joint_position_upper =
      std::make_shared<robotoc::JointPositionUpperLimit>(robot);
  auto joint_velocity_lower =
      std::make_shared<robotoc::JointVelocityLowerLimit>(robot);
  auto joint_velocity_upper =
      std::make_shared<robotoc::JointVelocityUpperLimit>(robot);
  auto joint_torques_lower =
      std::make_shared<robotoc::JointTorquesLowerLimit>(robot);
  auto joint_torques_upper =
      std::make_shared<robotoc::JointTorquesUpperLimit>(robot);
  auto friction_cone = std::make_shared<robotoc::FrictionCone>(robot);
  constraints->add("joint_position_lower", joint_position_lower);
  constraints->add("joint_position_upper", joint_position_upper);
  constraints->add("joint_velocity_lower", joint_velocity_lower);
  constraints->add("joint_velocity_upper", joint_velocity_upper);
  constraints->add("joint_torques_lower", joint_torques_lower);
  constraints->add("joint_torques_upper", joint_torques_upper);
  constraints->add("friction_cone", friction_cone);

  // Create the contact sequence
  auto contact_sequence = std::make_shared<robotoc::ContactSequence>(robot);
  const double mu = 0.7;
  const std::unordered_map<std::string, double> friction_coefficients = {
      {"FrontLeft_paw", mu},
      {"BackLeft_paw", mu},
      {"FrontRight_paw", mu},
      {"BackRight_paw", mu}};

  robot.updateFrameKinematics(q_standing);
  const Eigen::Vector3d x3d0_LF = robot.framePosition("FrontLeft_paw");
  const Eigen::Vector3d x3d0_LH = robot.framePosition("BackLeft_paw");
  const Eigen::Vector3d x3d0_RF = robot.framePosition("FrontRight_paw");
  const Eigen::Vector3d x3d0_RH = robot.framePosition("BackRight_paw");

  std::unordered_map<std::string, Eigen::Vector3d> contact_positions = {
      {"FrontLeft_paw", x3d0_LF},
      {"BackLeft_paw", x3d0_LH},
      {"FrontRight_paw", x3d0_RF},
      {"BackRight_paw", x3d0_RH}};
  auto contact_status_standing = robot.createContactStatus();
  contact_status_standing.activateContacts(std::vector<std::string>(
      {"FrontLeft_paw", "BackLeft_paw", "FrontRight_paw", "BackRight_paw"}));
  contact_status_standing.setContactPlacements(contact_positions);
  contact_status_standing.setFrictionCoefficients(friction_coefficients);
  contact_sequence->init(contact_status_standing);

  auto contact_status_flying = robot.createContactStatus();
  contact_sequence->push_back(contact_status_flying, t0 + ground_time - 0.3,
                              true);

  contact_positions["FrontLeft_paw"].noalias() += jump_length;
  contact_positions["BackLeft_paw"].noalias() += jump_length;
  contact_positions["FrontRight_paw"].noalias() += jump_length;
  contact_positions["BackRight_paw"].noalias() += jump_length;
  contact_status_standing.setContactPlacements(contact_positions);
  contact_sequence->push_back(contact_status_standing,
                              t0 + ground_time + flying_time - 0.1, true);

  // Create the STO cost function
  auto sto_cost = std::make_shared<robotoc::STOCostFunction>();
  // Create the STO constraints
  const std::vector<double> minimum_dwell_times = {0.15, 0.15, 0.65};
  auto sto_constraints = std::make_shared<robotoc::STOConstraints>(
      minimum_dwell_times, barrier_param, fraction_to_boundary_rule);

  // you can check the contact sequence via
  // std::cout << contact_sequence << std::endl;

  test_contact_dyn(robot, contact_status_standing);
  test_config_cost(robot, config_cost);

  return 0;
}