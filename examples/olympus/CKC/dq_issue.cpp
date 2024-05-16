#include <iostream>
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/kinematics-derivatives.hpp"
#include "pinocchio/algorithm/frames-derivatives.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/parsers/sample-models.hpp"



using namespace pinocchio;
using namespace Eigen;


int main()
{
  
  pinocchio::Model model;
  pinocchio::buildModels::humanoidRandom(model,true);
  pinocchio::Data data(model);
  const std::string FL_FOOT = "lleg6_body";

  model.lowerPositionLimit.head<3>().fill(-1.);
  model.upperPositionLimit.head<3>().fill( 1.);
  
  VectorXd q = randomConfiguration(model);
  VectorXd v = VectorXd::Random(model.nv);
  VectorXd a = VectorXd::Random(model.nv);

  for (pinocchio::ReferenceFrame rf : {pinocchio::LOCAL, pinocchio::LOCAL_WORLD_ALIGNED, pinocchio::WORLD})
  {
  // jacobians of velocity
  Eigen::MatrixXd d_v_dq = Eigen::MatrixXd::Zero(6,model.nv);
  Eigen::MatrixXd d_v_dv = Eigen::MatrixXd::Zero(6,model.nv);
  Eigen::MatrixXd d_v_dq_ref = Eigen::MatrixXd::Zero(3,model.nv);
  // jacobians of acceleration
  Eigen::MatrixXd d_a_dq = Eigen::MatrixXd::Zero(6,model.nv);
  Eigen::MatrixXd d_a_dv = Eigen::MatrixXd::Zero(6,model.nv);
  Eigen::MatrixXd d_a_da = Eigen::MatrixXd::Zero(6,model.nv);
  Eigen::MatrixXd d_a_dq_ref = Eigen::MatrixXd::Zero(3,model.nv);

  // get analytical derivatives  
  pinocchio::forwardKinematics(model,data,q,v,a);
  pinocchio::updateFramePlacements(model,data);
  pinocchio::computeJointJacobians(model, data, q);
  pinocchio::computeForwardKinematicsDerivatives(model,data,q,v,a);
  pinocchio::getFrameAccelerationDerivatives(model,data,model.getFrameId(FL_FOOT),rf,d_v_dq,d_a_dq,d_a_dv,d_a_da);
  pinocchio::getFrameVelocityDerivatives(model,data,model.getFrameId(FL_FOOT),rf,d_v_dq,d_v_dv);
  
  // get numerical derivatives
  Eigen::Vector3d vel = pinocchio::getFrameVelocity(model, data, model.getFrameId(FL_FOOT), rf).linear();
  Eigen::Vector3d acc = pinocchio::getFrameAcceleration(model, data, model.getFrameId(FL_FOOT), rf).linear();
  float eps = 1e-8;
  for(int i = 0; i < model.nv; i++)
  {
    Eigen::VectorXd q_next(model.nq);
    Eigen::VectorXd dq = Eigen::VectorXd::Zero(model.nv);
    dq(i) = eps;
    pinocchio::integrate(model,q,dq,q_next);
    pinocchio::forwardKinematics(model,data,q_next,v,a);
    pinocchio::updateFramePlacements(model,data);
    Eigen::Vector3d vel_next = pinocchio::getFrameVelocity(model, data, model.getFrameId(FL_FOOT), rf).linear();
    Eigen::Vector3d acc_next = pinocchio::getFrameAcceleration(model, data, model.getFrameId(FL_FOOT), rf).linear();
    d_v_dq_ref.col(i) = (vel_next - vel)/eps;
    d_a_dq_ref.col(i) = (acc_next - acc)/eps;
  }
  
  std::string rf_name;
  switch (rf)
  {
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
  std::cout << "dv_dq error: " << (d_v_dq.template topRows<3>() - d_v_dq_ref).norm() << std::endl;
  std::cout << "da_dq error: " << (d_a_dq.template topRows<3>() - d_a_dq_ref).norm() << std::endl;
  std::cout << "====================================" << std::endl;
  
  }
  
}
