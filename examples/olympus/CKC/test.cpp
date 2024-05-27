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
  Eigen::Vector2d baumgarte_residual = Eigen::Vector2d::Zero();
  Eigen::Vector2d buamgarte_residual_perturbed_q,
      buamgarte_residual_perturbed_v, buamgarte_residual_perturbed_a;
  Eigen::VectorXd q_perturbed(model.nq), v_perturbed(model.nv),
      a_perturbed(model.nv);
  float finite_diff = 1e-4;

  ckc.updateKinematics(q, v, a);
  ckc.computeBaumgarteResidual(baumgarte_residual);

  for (int i = 0; i < model.nv; i++) {
    Eigen::VectorXd dq = Eigen::VectorXd::Zero(model.nv);
    dq(i) = finite_diff;
    pinocchio::integrate(model, q, dq, q_perturbed);
    ckc.updateKinematics(q_perturbed, v, a);
    ckc.computeBaumgarteResidual(buamgarte_residual_perturbed_q);
    baumgarte_partial_dq_approx.col(i) =
        (buamgarte_residual_perturbed_q - baumgarte_residual) / finite_diff;
  }

  for (int i = 0; i < model.nv; i++) {
    v_perturbed = v;
    v_perturbed(i) += finite_diff;
    ckc.updateKinematics(q, v_perturbed, a);
    ckc.computeBaumgarteResidual(buamgarte_residual_perturbed_v);
    baumgarte_partial_dv_approx.col(i) =
        (buamgarte_residual_perturbed_v - baumgarte_residual) / finite_diff;
  }

  for (int i = 0; i < model.nv; i++) {
    a_perturbed = a;
    a_perturbed(i) += finite_diff;
    ckc.updateKinematics(q, v, a_perturbed);
    ckc.computeBaumgarteResidual(buamgarte_residual_perturbed_a);
    baumgarte_partial_da_approx.col(i) =
        (buamgarte_residual_perturbed_a - baumgarte_residual) / finite_diff;
  }
}

void test_finite_diff(pinocchio::Model &model, pinocchio::Data &data,
                      robotoc::CKC &ckc, Eigen::VectorXd &q, Eigen::VectorXd &v,
                      Eigen::VectorXd &a) {

  Eigen::MatrixXd baumgarte_partial_dq = Eigen::MatrixXd::Zero(2, model.nv);
  Eigen::MatrixXd baumgarte_partial_dv = Eigen::MatrixXd::Zero(2, model.nv);
  Eigen::MatrixXd baumgarte_partial_da = Eigen::MatrixXd::Zero(2, model.nv);

  Eigen::MatrixXd baumgarte_partial_dq_approx =
      Eigen::MatrixXd::Zero(2, model.nv);
  Eigen::MatrixXd baumgarte_partial_dv_approx =
      Eigen::MatrixXd::Zero(2, model.nv);
  Eigen::MatrixXd baumgarte_partial_da_approx =
      Eigen::MatrixXd::Zero(2, model.nv);

  ckc.updateKinematics(q, v, a);
  ckc.computeBaumgarteDerivatives(baumgarte_partial_dq, baumgarte_partial_dv,
                                  baumgarte_partial_da);

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

int main() {
  pinocchio::Model model;
  std::string urdf_file;

  const std::string FL_INNER("FrontLeft_ankle_inner");
  const std::string FL_OUTER("FrontLeft_ankle_outer");
  const std::string FR_INNER("FrontRight_ankle_inner");
  const std::string FR_OUTER("FrontRight_ankle_outer");
  const std::string BL_INNER("BackLeft_ankle_inner");
  const std::string BL_OUTER("BackLeft_ankle_outer");
  const std::string BR_INNER("BackRight_ankle_inner");
  const std::string BR_OUTER("BackRight_ankle_outer");

  urdf_file = std::string(
      "/home/bolivar/OLYMPOC/robotoc/descriptions/olympus_description/"
      "urdf/olympus.urdf");
  pinocchio::urdf::buildModel(urdf_file, JointModelFreeFlyer(), model);

  pinocchio::Data data(model);
  model.lowerPositionLimit.head<3>().fill(-3.);
  model.upperPositionLimit.head<3>().fill(3.);

  VectorXd q = VectorXd::Zero(model.nq);
  q = pinocchio::randomConfiguration(model);

  VectorXd v = VectorXd::Random(model.nv);
  VectorXd a = VectorXd::Random(model.nv);

  robotoc::CKCInfo info_FL(FL_INNER, FL_OUTER, 1, 1);
  robotoc::CKCInfo info_FR(FR_INNER, FR_OUTER, 1, 1);
  robotoc::CKCInfo info_BL(BL_INNER, BL_OUTER, 1, 1);
  robotoc::CKCInfo info_BR(BR_INNER, BR_OUTER, 1, 1);
  robotoc::CKC ckc_FL(model, info_FL);
  robotoc::CKC ckc_FR(model, info_FR);
  robotoc::CKC ckc_BL(model, info_BL);
  robotoc::CKC ckc_BR(model, info_BR);

  Eigen::Vector2d baumgarte_residual = Eigen::Vector2d::Zero();

  Eigen::MatrixXd baumgarte_partial_dq = Eigen::MatrixXd::Zero(2, model.nv);
  Eigen::MatrixXd baumgarte_partial_dv = Eigen::MatrixXd::Zero(2, model.nv);
  Eigen::MatrixXd baumgarte_partial_da = Eigen::MatrixXd::Zero(2, model.nv);

  std::vector<robotoc::CKC> ckcs = {ckc_FL, ckc_FR, ckc_BL, ckc_BR};

  for (auto &ckc : ckcs) {
    ckc.updateKinematics(q, v, a);
    ckc.computeBaumgarteResidual(baumgarte_residual);
    std::cout << "baumgarte_residual: " << baumgarte_residual.transpose()
              << std::endl;
    test_finite_diff(model, data, ckc, q, v, a);
  }

  return 0;
}
