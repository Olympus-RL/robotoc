#include "robotoc/robot/ckc.hpp"
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

void compute_finite_diff(pinocchio::Model &model, pinocchio::Data &data,
                         robotoc::CKC &ckc, Eigen::VectorXd &q,
                         Eigen::VectorXd &v, Eigen::VectorXd &a,
                         Eigen::MatrixXd &baumgarte_partial_dq_approx,
                         Eigen::MatrixXd &baumgarte_partial_dv_approx,
                         Eigen::MatrixXd &baumgarte_partial_da_approx) {
  Eigen::Vector3d baumgarte_residual = Eigen::Vector3d::Zero();
  Eigen::Vector3d buamgarte_residual_perturbed_q,
      buamgarte_residual_perturbed_v, buamgarte_residual_perturbed_a;
  Eigen::VectorXd q_perturbed(model.nq), v_perturbed(model.nv),
      a_perturbed(model.nv);
  float finite_diff = 1e-4;

  pinocchio::forwardKinematics(model, data, q, v, a);
  pinocchio::updateFramePlacements(model, data);
  ckc.computeBaumgarteResidual(model, data, baumgarte_residual);

  for (int i = 0; i < model.nv; i++) {
    Eigen::VectorXd dq = Eigen::VectorXd::Zero(model.nv);
    dq(i) = finite_diff;
    pinocchio::integrate(model, q, dq, q_perturbed);
    pinocchio::forwardKinematics(model, data, q_perturbed, v, a);
    pinocchio::updateFramePlacements(model, data);
    ckc.computeBaumgarteResidual(model, data, buamgarte_residual_perturbed_q);
    baumgarte_partial_dq_approx.col(i) =
        (buamgarte_residual_perturbed_q - baumgarte_residual) / finite_diff;
  }

  for (int i = 0; i < model.nv; i++) {
    v_perturbed = v;
    v_perturbed(i) += finite_diff;
    pinocchio::forwardKinematics(model, data, q, v_perturbed, a);
    pinocchio::updateFramePlacements(model, data);
    ckc.computeBaumgarteResidual(model, data, buamgarte_residual_perturbed_v);
    baumgarte_partial_dv_approx.col(i) =
        (buamgarte_residual_perturbed_v - baumgarte_residual) / finite_diff;
  }

  for (int i = 0; i < model.nv; i++) {
    a_perturbed = a;
    a_perturbed(i) += finite_diff;
    pinocchio::forwardKinematics(model, data, q, v, a_perturbed);
    pinocchio::updateFramePlacements(model, data);
    ckc.computeBaumgarteResidual(model, data, buamgarte_residual_perturbed_a);
    baumgarte_partial_da_approx.col(i) =
        (buamgarte_residual_perturbed_a - baumgarte_residual) / finite_diff;
  }
}

void test_finite_diff(pinocchio::Model &model, pinocchio::Data &data,
                      robotoc::CKC &ckc, Eigen::VectorXd &q, Eigen::VectorXd &v,
                      Eigen::VectorXd &a) {

  Eigen::MatrixXd baumgarte_partial_dq = Eigen::MatrixXd::Zero(3, model.nv);
  Eigen::MatrixXd baumgarte_partial_dv = Eigen::MatrixXd::Zero(3, model.nv);
  Eigen::MatrixXd baumgarte_partial_da = Eigen::MatrixXd::Zero(3, model.nv);

  Eigen::MatrixXd baumgarte_partial_dq_approx =
      Eigen::MatrixXd::Zero(3, model.nv);
  Eigen::MatrixXd baumgarte_partial_dv_approx =
      Eigen::MatrixXd::Zero(3, model.nv);
  Eigen::MatrixXd baumgarte_partial_da_approx =
      Eigen::MatrixXd::Zero(3, model.nv);

pinocchio:
  forwardKinematics(model, data, q, v, a);
  pinocchio::updateFramePlacements(model, data);
  pinocchio::computeForwardKinematicsDerivatives(model, data, q, v, a);
  ckc.computeBaumgarteDerivatives(model, data, baumgarte_partial_dq,
                                  baumgarte_partial_dv, baumgarte_partial_da);

  compute_finite_diff(model, data, ckc, q, v, a, baumgarte_partial_dq_approx,
                      baumgarte_partial_dv_approx, baumgarte_partial_da_approx);

  std::cout << "dq error: "
            << (baumgarte_partial_dq - baumgarte_partial_dq_approx).norm()
            << std::endl;
  std::cout << "dv error: "
            << (baumgarte_partial_dv - baumgarte_partial_dv_approx).norm()
            << std::endl;
  std::cout << "da error: "
            << (baumgarte_partial_da - baumgarte_partial_da_approx).norm()
            << std::endl;
}

void test_finite_diff_contact(pinocchio::Model &model, pinocchio::Data &data,
                              robotoc::PointContact &contact,
                              Eigen::VectorXd &q, Eigen::VectorXd &v,
                              Eigen::VectorXd &a) {

  Eigen::Vector3d desired_contact_position = Eigen::Vector3d::Zero();
pinocchio:
  forwardKinematics(model, data, q, v, a);
  pinocchio::updateFramePlacements(model, data);
  pinocchio::computeForwardKinematicsDerivatives(model, data, q, v, a);

  Eigen::Vector3d baumgarte_residual;
  Eigen::MatrixXd baumgarte_partial_dq = Eigen::MatrixXd::Zero(3, model.nv);
  Eigen::MatrixXd baumgarte_partial_dv = Eigen::MatrixXd::Zero(3, model.nv);
  Eigen::MatrixXd baumgarte_partial_da = Eigen::MatrixXd::Zero(3, model.nv);

  contact.computeBaumgarteResidual(model, data, desired_contact_position,
                                   baumgarte_residual);
  contact.computeBaumgarteDerivatives(model, data, baumgarte_partial_dq,
                                      baumgarte_partial_dv,
                                      baumgarte_partial_da);

  Eigen::Vector3d buamgarte_residual_perturbed_q,
      buamgarte_residual_perturbed_v, buamgarte_residual_perturbed_a;
  Eigen::MatrixXd baumgarte_partial_dq_approx =
      Eigen::MatrixXd::Zero(3, model.nv);
  Eigen::MatrixXd baumgarte_partial_dv_approx =
      Eigen::MatrixXd::Zero(3, model.nv);
  Eigen::MatrixXd baumgarte_partial_da_approx =
      Eigen::MatrixXd::Zero(3, model.nv);

  Eigen::VectorXd q_perturbed(model.nq);
  Eigen::VectorXd v_perturbed(model.nv);
  Eigen::VectorXd a_perturbed(model.nv);

  float finite_diff = 1e-4;

  for (int i = 0; i < model.nv; i++) {
    Eigen::VectorXd dq = Eigen::VectorXd::Zero(model.nv);
    dq(i) = finite_diff;
    pinocchio::integrate(model, q, dq, q_perturbed);
    pinocchio::forwardKinematics(model, data, q_perturbed, v, a);
    pinocchio::updateFramePlacements(model, data);
    contact.computeBaumgarteResidual(model, data, desired_contact_position,
                                     buamgarte_residual_perturbed_q);
    baumgarte_partial_dq_approx.col(i) =
        (buamgarte_residual_perturbed_q - baumgarte_residual) / finite_diff;
  }

  for (int i = 0; i < model.nv; i++) {
    v_perturbed = v;
    v_perturbed(i) += finite_diff;
    pinocchio::forwardKinematics(model, data, q, v_perturbed, a);
    pinocchio::updateFramePlacements(model, data);
    contact.computeBaumgarteResidual(model, data, desired_contact_position,
                                     buamgarte_residual_perturbed_v);
    baumgarte_partial_dv_approx.col(i) =
        (buamgarte_residual_perturbed_v - baumgarte_residual) / finite_diff;
  }

  for (int i = 0; i < model.nv; i++) {
    a_perturbed = a;
    a_perturbed(i) += finite_diff;
    pinocchio::forwardKinematics(model, data, q, v, a_perturbed);
    pinocchio::updateFramePlacements(model, data);
    contact.computeBaumgarteResidual(model, data, desired_contact_position,
                                     buamgarte_residual_perturbed_a);
    baumgarte_partial_da_approx.col(i) =
        (buamgarte_residual_perturbed_a - baumgarte_residual) / finite_diff;
  }

  std::cout << "dq error: "
            << (baumgarte_partial_dq - baumgarte_partial_dq_approx).norm()
            << std::endl;
  std::cout << "dv error: "
            << (baumgarte_partial_dv - baumgarte_partial_dv_approx).norm()
            << std::endl;
  std::cout << "da error: "
            << (baumgarte_partial_da - baumgarte_partial_da_approx).norm()
            << std::endl;
}

int main() {
  const int MODEL_NAME = 2;
  std::string FR_FOOT, FL_FOOT;
  pinocchio::Model model;
  std::string urdf_file;

  // const std::string
  // urdf_file("/home/bolivar/OLYMPOC/robotoc/descriptions/olympus_description/urdf/olympus.urdf");

  switch (MODEL_NAME) {
  case 0:
    FR_FOOT = "RF_FOOT";
    FL_FOOT = "LF_FOOT";
    urdf_file = std::string("/home/bolivar/OLYMPOC/robotoc/examples/anymal/"
                            "anymal_b_simple_description/urdf/anymal.urdf");
    pinocchio::urdf::buildModel(urdf_file, JointModelFreeFlyer(), model);
    break;
  case 1:
    FR_FOOT = "rleg6_body";
    FL_FOOT = "lleg6_body";
    pinocchio::buildModels::humanoidRandom(model, true);
    break;

  case 2:
    FR_FOOT = "FrontLeft_ankle_inner";
    FL_FOOT = "FrontLeft_ankle_outer";
    urdf_file = std::string(
        "/home/bolivar/OLYMPOC/robotoc/descriptions/olympus_description/"
        "urdf/olympus.urdf");
    pinocchio::urdf::buildModel(urdf_file, JointModelFreeFlyer(), model);

    break;
  default:
    std::cout << "Model not found" << std::endl;
    return 0;
  }
  pinocchio::Data data(model);
  model.lowerPositionLimit.head<3>().fill(-1.);
  model.upperPositionLimit.head<3>().fill(1.);

  VectorXd q = randomConfiguration(model);
  VectorXd v = VectorXd::Random(model.nv);
  VectorXd a = VectorXd::Random(model.nv);

  robotoc::CKCInfo info(FR_FOOT, FL_FOOT, 1, 1);
  robotoc::CKC ckc(model, info);
  robotoc::ContactModelInfo contact_info(FL_FOOT, 1, 1);
  robotoc::PointContact contact(model, contact_info);

  test_finite_diff(model, data, ckc, q, v, a);
  test_finite_diff_contact(model, data, contact, q, v, a);

  return 0;
}
