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
#include "robotoc/cost/trajectory_ref.hpp"
#include "robotoc/ocp/ocp.hpp"
#include "robotoc/planner/contact_sequence.hpp"
#include "robotoc/robot/robot.hpp"
#include "robotoc/solver/ocp_solver.hpp"
#include "robotoc/solver/solver_options.hpp"
#include "robotoc/sto/sto_constraints.hpp"
#include "robotoc/sto/sto_cost_function.hpp"

int main(int argc, char *argv[]) {
  robotoc::RobotModelInfo model_info;
  model_info.urdf_path = "/home/bolivar/OLYMPOC/robotoc/descriptions/"
                         "olympus_description/urdf/olympus_open_scaled.urdf";
  model_info.base_joint_type = robotoc::BaseJointType::FloatingBase;
  const double baumgarte_time_step = 0.05;
  model_info.point_contacts = {
      robotoc::ContactModelInfo("FrontLeft_paw", baumgarte_time_step),
      robotoc::ContactModelInfo("BackLeft_paw", baumgarte_time_step),
      robotoc::ContactModelInfo("FrontRight_paw", baumgarte_time_step),
      robotoc::ContactModelInfo("BackRight_paw", baumgarte_time_step)};
  // model_info.ckcs = {
  //    robotoc::CKCInfo("FrontLeft_paw", "BackLeft_paw", baumgarte_time_step)};
  robotoc::Robot robot(model_info);
  // robot.setGravity(-3.72);

  robot.setLowerJointPositionLimit(
      Eigen::VectorXd::Constant(robot.dimv() - 6, -1.57));
  robot.setUpperJointPositionLimit(
      Eigen::VectorXd::Constant(robot.dimv() - 6, 1.57));
  robot.setJointEffortLimit(Eigen::VectorXd::Constant(robot.dimv() - 6, 80));
  robot.setJointVelocityLimit(
      Eigen::VectorXd::Constant(robot.dimv() - 6, 31.0));

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
  q_standing(2) = 0.45;
  q_standing(6) = 1.0;
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
  robotoc::TrajectoryRef::Traj ref_traj;
  ref_traj.push_back(robotoc::TrajectoryRef::StageTraj(10, q_ref));
  ref_traj.push_back(robotoc::TrajectoryRef::StageTraj(10, q_ref));
  ref_traj.push_back(robotoc::TrajectoryRef::StageTraj(10, q_ref));
  auto ref = std::make_shared<robotoc::TrajectoryRef>(robot, ref_traj);
  config_cost->set_ref(ref);
  config_cost->set_q_weight(q_weight);
  config_cost->set_q_weight_terminal(q_weight);
  //config_cost->set_q_weight_impact(q_weight_impact);
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
  const double mu = 1.7;
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
  contact_sequence->push_back(contact_status_flying, t0 + ground_time, true);

  contact_positions["FrontLeft_paw"].noalias() += jump_length;
  contact_positions["BackLeft_paw"].noalias() += jump_length;
  contact_positions["FrontRight_paw"].noalias() += jump_length;
  contact_positions["BackRight_paw"].noalias() += jump_length;
  contact_status_standing.setContactPlacements(contact_positions);
  contact_sequence->push_back(contact_status_standing,
                              t0 + ground_time + flying_time, true);

  // Create the STO cost function
  auto sto_cost = std::make_shared<robotoc::STOCostFunction>();
  // Create the STO constraints
  const std::vector<double> minimum_dwell_times = {0.15, 0.15, 0.65};
  auto sto_constraints = std::make_shared<robotoc::STOConstraints>(
      minimum_dwell_times, barrier_param, fraction_to_boundary_rule);

  // you can check the contact sequence via
  // std::cout << contact_sequence << std::endl;

  const double T = t0 + flying_time + 2 * ground_time;
  const int N = std::floor(T / dt);
  robotoc::OCP ocp(robot, cost, constraints, sto_cost, sto_constraints,
                   contact_sequence, T, N);
  auto solver_options = robotoc::SolverOptions();
  solver_options.max_dt_mesh = T / N;
  solver_options.kkt_tol_mesh = 0.1;
  solver_options.max_iter = 800;
  solver_options.nthreads = 4;
  solver_options.enable_benchmark = true;
  solver_options.initial_sto_reg_iter = 10;
  solver_options.enable_line_search = false;
  robotoc::OCPSolver ocp_solver(ocp, solver_options);

  // Initial time and initial state
  const double t = 0;
  Eigen::VectorXd q(q_standing);
  Eigen::VectorXd v(Eigen::VectorXd::Zero(robot.dimv()));

  // Solves the OCP.
  ocp_solver.discretize(t);
  ocp_solver.setSolution("q", q);
  ocp_solver.setSolution("v", v);
  Eigen::Vector3d f_init;
  f_init << 0, 0, 0.25 * robot.totalWeight();
  ocp_solver.setSolution("f", f_init);
  ocp_solver.discretize(t);
  ocp_solver.initConstraints();
  std::cout << "3" << std::endl;
  std::cout << "Initial KKT error: " << ocp_solver.KKTError(t, q, v)
            << std::endl;
  std::cout << "4" << std::endl;
  ocp_solver.solve(t, q, v);
  std::cout << "KKT error after convergence: " << ocp_solver.KKTError(t, q, v)
            << std::endl;
  std::cout << ocp_solver.getSolverStatistics() << std::endl;

  // const int num_iteration = 10000;
  // robotoc::benchmark::CPUTime(ocp_solver, t, q, v, num_iteration);

  return 0;
}
