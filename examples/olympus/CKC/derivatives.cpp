#include "robotoc/robot/ckc.hxx"
#include "robotoc/robot/point_contact.hpp"

#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/parsers/sample-models.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include <iostream>

using namespace pinocchio;
using namespace Eigen;
using namespace robotoc;

int main() {
  const int MODEL_NAME = 1;
  std::string FR_FOOT, FL_FOOT;
  pinocchio::Model model;
  std::string urdf_file;

  // const std::string
  // urdf_file("/home/bolivar/OLYMPOC/robotoc/descriptions/olympus_description/urdf/olympus.urdf");

  switch (MODEL_NAME) {
  case 0:
    FR_FOOT = "BackRight_ankle_outer";
    FL_FOOT = "BackRight_ankle_inner";
    urdf_file = std::string("/home/bolivar/OLYMPOC/robotoc/examples/anymal/"
                            "anymal_b_simple_description/urdf/anymal.urdf");
    pinocchio::urdf::buildModel(urdf_file, model);
    break;
  case 1:
    FR_FOOT = "rleg6_body";
    FL_FOOT = "lleg6_body";
    pinocchio::buildModels::humanoidRandom(model, true);
    break;
  default:
    std::cout << "Model not found" << std::endl;
    return 0;
  }

  const pinocchio::ReferenceFrame rf = pinocchio::LOCAL;
  pinocchio::Data data(model);
  model.lowerPositionLimit.head<3>().fill(-1.);
  model.upperPositionLimit.head<3>().fill(1.);

  VectorXd q = randomConfiguration(model);
  VectorXd v = VectorXd::Random(model.nv);
  VectorXd a = VectorXd::Random(model.nv);

  for (pinocchio::ReferenceFrame rf :
       {pinocchio::LOCAL, pinocchio::LOCAL_WORLD_ALIGNED, pinocchio::WORLD}) {

    Eigen::MatrixXd d_v_dq = Eigen::MatrixXd::Zero(6, model.nv);
    Eigen::MatrixXd d_v_dv = Eigen::MatrixXd::Zero(6, model.nv);
    Eigen::MatrixXd d_v_dv_ref = Eigen::MatrixXd::Zero(3, model.nv);
    Eigen::MatrixXd d_v_dq_ref = Eigen::MatrixXd::Zero(3, model.nv);
    Eigen::MatrixXd d_a_dq = Eigen::MatrixXd::Zero(6, model.nv);
    Eigen::MatrixXd d_a_dv = Eigen::MatrixXd::Zero(6, model.nv);
    Eigen::MatrixXd d_a_da = Eigen::MatrixXd::Zero(6, model.nv);
    Eigen::MatrixXd d_a_dq_ref = Eigen::MatrixXd::Zero(3, model.nv);
    Eigen::MatrixXd d_a_dv_ref = Eigen::MatrixXd::Zero(3, model.nv);
    Eigen::MatrixXd d_a_da_ref = Eigen::MatrixXd::Zero(3, model.nv);

    pinocchio::forwardKinematics(model, data, q, v, a);
    pinocchio::updateFramePlacements(model, data);
    pinocchio::computeJointJacobians(model, data, q);
    pinocchio::computeForwardKinematicsDerivatives(model, data, q, v, a);

    pinocchio::getFrameAccelerationDerivatives(model, data,
                                               model.getFrameId(FL_FOOT), rf,
                                               d_v_dq, d_a_dq, d_a_dv, d_a_da);
    pinocchio::getFrameVelocityDerivatives(
        model, data, model.getFrameId(FL_FOOT), rf, d_v_dq, d_v_dv);

    Eigen::Vector3d vel =
        pinocchio::getFrameVelocity(model, data, model.getFrameId(FL_FOOT), rf)
            .linear();
    Eigen::Vector3d acc = pinocchio::getFrameAcceleration(
                              model, data, model.getFrameId(FL_FOOT), rf)
                              .linear();

    float eps = 1e-8;
    for (int i = 0; i < model.nv; i++) {
      Eigen::VectorXd q_next(model.nq);
      Eigen::VectorXd dq = Eigen::VectorXd::Zero(model.nv);
      dq(i) = eps;
      pinocchio::integrate(model, q, dq, q_next);
      pinocchio::forwardKinematics(model, data, q_next, v, a);
      pinocchio::updateFramePlacements(model, data);
      Eigen::Vector3d vel_next = pinocchio::getFrameVelocity(
                                     model, data, model.getFrameId(FL_FOOT), rf)
                                     .linear();
      Eigen::Vector3d acc_next = pinocchio::getFrameAcceleration(
                                     model, data, model.getFrameId(FL_FOOT), rf)
                                     .linear();
      d_v_dq_ref.col(i) = (vel_next - vel) / eps;
      d_a_dq_ref.col(i) = (acc_next - acc) / eps;
    }

    for (int i = 0; i < model.nv; i++) {
      Eigen::VectorXd v_next = v;
      v_next(i) += eps;
      pinocchio::forwardKinematics(model, data, q, v_next, a);
      pinocchio::updateFramePlacements(model, data);
      Eigen::Vector3d vel_next = pinocchio::getFrameVelocity(
                                     model, data, model.getFrameId(FL_FOOT), rf)
                                     .linear();
      Eigen::Vector3d acc_next = pinocchio::getFrameAcceleration(
                                     model, data, model.getFrameId(FL_FOOT), rf)
                                     .linear();
      d_v_dv_ref.col(i) = (vel_next - vel) / eps;
      d_a_dv_ref.col(i) = (acc_next - acc) / eps;
    }

    for (int i = 0; i < model.nv; i++) {
      Eigen::VectorXd a_next = a;
      a_next(i) += eps;
      pinocchio::forwardKinematics(model, data, q, v, a_next);
      pinocchio::updateFramePlacements(model, data);
      Eigen::Vector3d acc_next = pinocchio::getFrameAcceleration(
                                     model, data, model.getFrameId(FL_FOOT), rf)
                                     .linear();
      d_a_da_ref.col(i) = (acc_next - acc) / eps;
    }

    std::string rf_name;
    switch (rf) {
    case pinocchio::LOCAL:
      rf_name = "LOCAL";
      break;
    case pinocchio::LOCAL_WORLD_ALIGNED:
      rf_name = "LOCAL_WORLD_ALIGNED";
      break;
    case pinocchio::WORLD:
      rf_name = "WORLD";
      break;
    }

    std::cout << "Frame derivative test: " << rf_name << std::endl;
    std::cout << "dv_dq error: "
              << (d_v_dq.template topRows<3>() - d_v_dq_ref).norm()
              << std::endl;
    std::cout << "dv_dv error: "
              << (d_v_dv.template topRows<3>() - d_v_dv_ref).norm()
              << std::endl;
    std::cout << "da_dq error: "
              << (d_a_dq.template topRows<3>() - d_a_dq_ref).norm()
              << std::endl;
    std::cout << "da_dv error: "
              << (d_a_dv.template topRows<3>() - d_a_dv_ref).norm()
              << std::endl;
    std::cout << "da_da error: "
              << (d_a_da.template topRows<3>() - d_a_da_ref).norm()
              << std::endl;
    std::cout << "====================================" << std::endl;
  }
}
