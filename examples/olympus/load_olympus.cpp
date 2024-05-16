#include "pinocchio/algorithm/aba.hpp"
#include "pinocchio/algorithm/centroidal.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/utils/timer.hpp"
#include <iostream>

using namespace pinocchio;
using namespace Eigen;

int main() {
  const std::string urdf_file("/home/bolivar/OLYMPOC/robotoc/descriptions/"
                              "olympus_description/urdf/olympus.urdf");

  pinocchio::Model model;
  pinocchio::urdf::buildModel(urdf_file, JointModelFreeFlyer(), model);
  pinocchio::Data data(model), data_ref(model);

  std::cout << "Model has " << model.nq << " joints and " << model.nv
            << " degrees of freedom" << std::endl;

  VectorXd q = VectorXd::Zero(model.nq);
  q(6) = 1.0;
  // Eigen::VectorXd q_lim = Eigen::VectorXd::Ones(model.nq);
  // pinocchio::randomConfiguration(model,-q_lim,q_lim,q);
  VectorXd v = VectorXd::Random(model.nv);
  VectorXd a = VectorXd::Random(model.nv);

  const std::string FL_ANKLE_INNER = "FrontLeft_ankle_inner";
  const std::string FL_ANKLE_OUTER = "FrontLeft_ankle_inner";
  const std::string FL_SHANK_INNER = "FrontLeft_shank_inner";

  const int FL_ANKLE_INNER_IDX = model.getFrameId(FL_ANKLE_INNER);
  const int FL_ANKLE_OUTER_IDX = model.getFrameId(FL_ANKLE_OUTER);

pinocchio:
  forwardKinematics(model, data, q, v, a);
  pinocchio::updateFramePlacements(model, data);
  pinocchio::computeJointJacobians(model, data);

  pinocchio::SE3 oMf = data.oMf[FL_ANKLE_INNER_IDX];
  Eigen::Vector3d vel_local_aligned =
      pinocchio::getFrameVelocity(model, data, FL_ANKLE_INNER_IDX,
                                  pinocchio::LOCAL_WORLD_ALIGNED)
          .linear();
  Eigen::Vector3d vel_world =
      pinocchio::getFrameVelocity(model, data, FL_ANKLE_INNER_IDX,
                                  pinocchio::WORLD)
          .linear();
  Eigen::Vector3d vel_local =
      pinocchio::getFrameVelocity(model, data, FL_ANKLE_INNER_IDX,
                                  pinocchio::LOCAL)
          .linear();
  Eigen::Vector3d p = data.oMf[FL_ANKLE_INNER_IDX].translation();
  Eigen::Vector3d acc =
      pinocchio::getFrameAcceleration(model, data, FL_ANKLE_INNER_IDX,
                                      pinocchio::LOCAL_WORLD_ALIGNED)
          .linear();
  acc = pinocchio::getFrameClassicalAcceleration(
            model, data, FL_ANKLE_INNER_IDX, pinocchio::LOCAL)
            .linear();
  Eigen::MatrixXd J_local = Eigen::MatrixXd(6, model.nv);
  Eigen::MatrixXd J_world = Eigen::MatrixXd(6, model.nv);
  Eigen::MatrixXd J_local_world_aligned = Eigen::MatrixXd(6, model.nv);
  pinocchio::getFrameJacobian(model, data, FL_ANKLE_INNER_IDX, pinocchio::LOCAL,
                              J_local);
  pinocchio::getFrameJacobian(model, data, FL_ANKLE_INNER_IDX, pinocchio::WORLD,
                              J_world);
  pinocchio::getFrameJacobian(model, data, FL_ANKLE_INNER_IDX,
                              pinocchio::LOCAL_WORLD_ALIGNED,
                              J_local_world_aligned);

  float eps = 1e-4;

  Eigen::VectorXd q_next(model.nq);
  pinocchio::integrate(model, q, eps * v, q_next);
  pinocchio::forwardKinematics(model, data_ref, q_next, v, a);
  pinocchio::updateFramePlacements(model, data_ref);

  Eigen::Vector3d p_next = data_ref.oMf[FL_ANKLE_INNER_IDX].translation();
  Eigen::Vector3d v_next =
      pinocchio::getFrameVelocity(model, data_ref, FL_ANKLE_INNER_IDX,
                                  pinocchio::LOCAL_WORLD_ALIGNED)
          .linear();

  Eigen::Vector3d v_est = (p_next - p) / eps;
  Eigen::Vector3d acc_est = (v_next - vel_local_aligned) / eps;

  std::cout << "v_local_aligned: " << vel_local_aligned.transpose()
            << std::endl;
  std::cout << "v_world: " << vel_world.transpose() << std::endl;
  std::cout << "v_local: " << vel_local.transpose() << std::endl;
  std::cout << "R*v_local_aligned: "
            << (oMf.rotation() * vel_local_aligned).transpose() << std::endl;
  std::cout << "J_local_v: " << (J_local * v).transpose() << std::endl;
  std::cout << "J_world_v: " << (J_world * v).transpose() << std::endl;
  std::cout << "J_local_world_aligned_v: "
            << (J_local_world_aligned * v).transpose() << std::endl;
  std::cout << "v_est: " << v_est.transpose() << std::endl;
  std::cout << "a: " << acc.transpose() << std::endl;
  std::cout << "a_est: " << acc_est.transpose() << std::endl;
}
