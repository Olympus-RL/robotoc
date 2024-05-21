#include "robotoc/dynamics/impact_state_equation.hpp"
#include "robotoc/dynamics/state_equation.hpp"

#include <cassert>

namespace robotoc {

void evalImpactStateEquation(const Robot& robot, const SplitSolution& s, 
                             const Eigen::VectorXd& q_next, 
                             const Eigen::VectorXd& v_next, 
                             SplitKKTResidual& kkt_residual) {
  assert(q_next.size() == robot.dimq());
  assert(v_next.size() == robot.dimv());
  robot.subtractConfiguration(s.q, q_next, kkt_residual.Fq());
  kkt_residual.Fv() = s.v + s.dv - v_next;
}


void evalImpactStateEquation(const Robot& robot, const SplitSolution& s, 
                             const SplitSolution& s_next, 
                             SplitKKTResidual& kkt_residual) {
  evalImpactStateEquation(robot, s, s_next.q, s_next.v, kkt_residual);
}


void linearizeImpactStateEquation(const Robot& robot, 
                                  const Eigen::VectorXd& q_prev, 
                                  const SplitSolution& s, 
                                  const SplitSolution& s_next, 
                                  StateEquationData& data, 
                                  SplitKKTMatrix& kkt_matrix, 
                                  SplitKKTResidual& kkt_residual) {
  assert(q_prev.size() == robot.dimq());
  evalImpactStateEquation(robot, s, s_next, kkt_residual);
  if (robot.hasFloatingBase()) {
    robot.dSubtractConfiguration_dqf(s.q, s_next.q, kkt_matrix.Fqq());
    data.Fqq_prev.setZero();
    robot.dSubtractConfiguration_dq0(q_prev, s.q, data.Fqq_prev);
    kkt_residual.lq().template head<6>().noalias() 
        += kkt_matrix.Fqq().template topLeftCorner<6, 6>().transpose() 
              * s_next.lmd.template head<6>();
    kkt_residual.lq().template head<6>().noalias() 
        += data.Fqq_prev.template topLeftCorner<6, 6>().transpose() 
              * s.lmd.template head<6>();
    kkt_residual.lq().tail(robot.dimv()-6).noalias() 
        += s_next.lmd.tail(robot.dimv()-6) - s.lmd.tail(robot.dimv()-6);
  }
  else {
    kkt_matrix.Fqq().diagonal().fill(1.);
    kkt_residual.lq().noalias() += s_next.lmd - s.lmd;
  }
  kkt_residual.lv().noalias() += s_next.gmm - s.gmm;
  kkt_residual.ldv.noalias() += s_next.gmm;
}


void correctLinearizeImpactStateEquation(const Robot& robot, 
                                         const SplitSolution& s, 
                                         const SplitSolution& s_next, 
                                         StateEquationData& data, 
                                         SplitKKTMatrix& kkt_matrix, 
                                         SplitKKTResidual& kkt_residual) {
  return;                                        
  if (!data.hasFloatingBase()) return;


  data.se3_jac_inverse.compute(data.Fqq_prev, data.Fqq_prev_inv);
  robot.dSubtractConfiguration_dq0(s.q, s_next.q, data.Fqq_prev);
  data.se3_jac_inverse.compute(data.Fqq_prev, data.Fqq_inv);
  data.Fqq_tmp = kkt_matrix.Fqq().template topLeftCorner<6, 6>();
  data.Fq_tmp  = kkt_residual.Fq().template head<6>();
  kkt_matrix.Fqq().template topLeftCorner<6, 6>().noalias() = - data.Fqq_inv * data.Fqq_tmp;
  kkt_residual.Fq().template head<6>().noalias() = - data.Fqq_inv * data.Fq_tmp;
}

} // namespace robotoc 