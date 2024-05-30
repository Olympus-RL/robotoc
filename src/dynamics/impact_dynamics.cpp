#include "robotoc/dynamics/impact_dynamics.hpp"

#include <cassert>

namespace robotoc {

void evalImpactDynamics(Robot &robot, const ImpactStatus &impact_status,
                        const SplitSolution &s, ContactDynamicsData &data) {
  data.setContactDimension(impact_status.dimf());
  data.setCKCDimension(0); // No CKC forces in impact dynamics
  robot.setImpactForces(impact_status, s.f_contact);
  robot.RNEAImpact(s.q, s.dv, data.ID_full());
  robot.computeImpactVelocityResidual(impact_status, data.C());
}

void linearizeImpactDynamics(Robot &robot, const ImpactStatus &impact_status,
                             const SplitSolution &s, ContactDynamicsData &data,
                             SplitKKTResidual &kkt_residual) {
  evalImpactDynamics(robot, impact_status, s, data);
  robot.RNEAImpactDerivatives(s.q, s.dv, data.dIDdq(), data.dIDddv);
  robot.computeImpactVelocityDerivatives(impact_status, data.dCdq(),
                                         data.dCdv());
  // augment inverse impact dynamics constraint
  kkt_residual.lq().noalias() += data.dIDdq().transpose() * s.beta;
  kkt_residual.ldv.noalias() += data.dIDddv.transpose() * s.beta;
  kkt_residual.lf().noalias() -= data.dCdv() * s.beta;
  // augment impact velocity constraint
  kkt_residual.lq().noalias() += data.dCdq().transpose() * s.mu_stack();
  kkt_residual.lv().noalias() += data.dCdv().transpose() * s.mu_stack();
  kkt_residual.ldv.noalias() += data.dCdv().transpose() * s.mu_stack();
  if (data.hasFloatingBase()) {
    data.lu_passive.setZero();
  }
}

void condenseImpactDynamics(Robot &robot, const ImpactStatus &impact_status,
                            ContactDynamicsData &data,
                            SplitKKTMatrix &kkt_matrix,
                            SplitKKTResidual &kkt_residual) {
  robot.computeMJtJinv(data.dIDddv, data.dCdv(), data.MJtJinv());
  const int dimv = robot.dimv();
  const int dimf = data.dimf();
  data.MJtJinv_dIDCdqv().leftCols(dimv).noalias() =
      data.MJtJinv() * data.dIDCdqv().leftCols(dimv);
  data.MJtJinv_dIDCdqv().topRightCorner(dimv, dimv).noalias() =
      data.MJtJinv().topRightCorner(dimv, dimf) * data.dCdv();
  data.MJtJinv_dIDCdqv().bottomRightCorner(dimf, dimv).noalias() =
      data.MJtJinv().bottomRightCorner(dimf, dimf) * data.dCdv();
  data.MJtJinv_IDC().noalias() = data.MJtJinv() * data.IDC();

  data.Qdvfqv().topRows(dimv).noalias() =
      (-kkt_matrix.Qdvdv.diagonal()).asDiagonal() *
      data.MJtJinv_dIDCdqv().topRows(dimv);
  data.Qdvfqv().bottomRows(dimf).noalias() =
      -kkt_matrix.Qff() * data.MJtJinv_dIDCdqv().bottomRows(dimf);
  data.Qdvfqv().bottomLeftCorner(dimf, dimv).noalias() -=
      kkt_matrix.Qqf().transpose();
  data.ldv() = kkt_residual.ldv;
  data.lf() = -kkt_residual.lf();
  data.ldv().noalias() -=
      kkt_matrix.Qdvdv.diagonal().asDiagonal() * data.MJtJinv_IDC().head(dimv);
  data.lf().noalias() -= kkt_matrix.Qff() * data.MJtJinv_IDC().tail(dimf);

  kkt_matrix.Qxx.noalias() -=
      data.MJtJinv_dIDCdqv().transpose() * data.Qdvfqv();
  kkt_matrix.Qxx.topRows(dimv).noalias() +=
      kkt_matrix.Qqf() * data.MJtJinv_dIDCdqv().bottomRows(dimf);
  kkt_residual.lx.noalias() -= data.MJtJinv_dIDCdqv().transpose() * data.ldvf();
  kkt_residual.lq().noalias() +=
      kkt_matrix.Qqf() * data.MJtJinv_IDC().tail(dimf);

  kkt_matrix.Fvq() = -data.MJtJinv_dIDCdqv().topLeftCorner(dimv, dimv);
  kkt_matrix.Fvv() = Eigen::MatrixXd::Identity(dimv, dimv) -
                     data.MJtJinv_dIDCdqv().topRightCorner(dimv, dimv);
  kkt_residual.Fv().noalias() -= data.MJtJinv_IDC().head(dimv);
}

void expandImpactDynamicsPrimal(const ContactDynamicsData &data,
                                SplitDirection &d) {
  d.ddvf().noalias() = -data.MJtJinv_dIDCdqv() * d.dx;
  d.ddvf().noalias() -= data.MJtJinv_IDC();
  d.df().array() *= -1;
}

void expandImpactDynamicsDual(ContactDynamicsData &data,
                              const SplitDirection &d_next, SplitDirection &d) {
  data.ldvf().noalias() += data.Qdvfqv() * d.dx;
  data.ldv().noalias() += d_next.dgmm();
  d.dbetamu().noalias() = -data.MJtJinv() * data.ldvf();
}

} // namespace robotoc