#include "robotoc/riccati/riccati_factorizer.hpp"

#include <cassert>

namespace robotoc {

RiccatiFactorizer::RiccatiFactorizer(const Robot &robot, const double max_dts0)
    : has_floating_base_(robot.hasFloatingBase()), dimv_(robot.dimv()),
      dimu_(robot.dimu()), max_dts0_(max_dts0),
      eps_(std::sqrt(std::numeric_limits<double>::epsilon())),
      llt_(robot.dimu()), llt_s_(), backward_recursion_(robot),
      c_riccati_(robot) {}

RiccatiFactorizer::RiccatiFactorizer()
    : has_floating_base_(false), dimv_(0), dimu_(0), max_dts0_(0), eps_(0),
      llt_(), llt_s_(), backward_recursion_(), c_riccati_() {}

RiccatiFactorizer::~RiccatiFactorizer() {}

void RiccatiFactorizer::setRegularization(const double max_dts0) {
  assert(max_dts0 > 0);
  max_dts0_ = max_dts0;
}

void RiccatiFactorizer::backwardRiccatiRecursion(
    const SplitRiccatiFactorization &riccati_next, SplitKKTMatrix &kkt_matrix,
    SplitKKTResidual &kkt_residual, SplitRiccatiFactorization &riccati,
    LQRPolicy &lqr_policy) {
  backward_recursion_.factorizeKKTMatrix(riccati_next, kkt_matrix,
                                         kkt_residual);
  llt_.compute(kkt_matrix.Quu);
  assert(llt_.info() == Eigen::Success);
  assert(kkt_matrix.dims() == kkt_residual.dims());
  riccati.setConstraintDimension(kkt_matrix.dims());
  c_riccati_.setConstraintDimension(kkt_matrix.dims());
  if (kkt_matrix.dims() == 0) {
    lqr_policy.K.noalias() = -llt_.solve(kkt_matrix.Qxu.transpose());
    lqr_policy.k.noalias() = -llt_.solve(kkt_residual.lu);
  } else {
    // Schur complement
    c_riccati_.Ginv.noalias() =
        llt_.solve(Eigen::MatrixXd::Identity(dimu_, dimu_));
    c_riccati_.DGinv().transpose().noalias() =
        llt_.solve(kkt_matrix.Phiu().transpose());
    c_riccati_.S().noalias() =
        c_riccati_.DGinv() * kkt_matrix.Phiu().transpose();
    llt_s_.compute(c_riccati_.S());
    assert(llt_s_.info() == Eigen::Success);
    c_riccati_.SinvDGinv().noalias() = llt_s_.solve(c_riccati_.DGinv());
    c_riccati_.Ginv.noalias() -=
        c_riccati_.SinvDGinv().transpose() * c_riccati_.DGinv();
    lqr_policy.K.noalias() = -c_riccati_.Ginv * kkt_matrix.Qxu.transpose();
    lqr_policy.K.noalias() -=
        c_riccati_.SinvDGinv().transpose() * kkt_matrix.Phix();
    lqr_policy.k.noalias() = -c_riccati_.Ginv * kkt_residual.lu;
    lqr_policy.k.noalias() -=
        c_riccati_.SinvDGinv().transpose() * kkt_residual.P();
    riccati.M().noalias() = llt_s_.solve(kkt_matrix.Phix());
    riccati.M().noalias() -=
        c_riccati_.SinvDGinv() * kkt_matrix.Qxu.transpose();
    riccati.m().noalias() = llt_s_.solve(kkt_residual.P());
    riccati.m().noalias() -= c_riccati_.SinvDGinv() * kkt_residual.lu;
    assert(!riccati.M().hasNaN());
    assert(!riccati.m().hasNaN());
  }
  assert(!lqr_policy.K.hasNaN());
  assert(!lqr_policy.k.hasNaN());
  backward_recursion_.factorizeRiccatiFactorization(
      riccati_next, kkt_matrix, kkt_residual, lqr_policy, riccati);
  if (kkt_matrix.dims() > 0) {
    c_riccati_.DtM.noalias() = kkt_matrix.Phiu().transpose() * riccati.M();
    c_riccati_.KtDtM.noalias() = lqr_policy.K.transpose() * c_riccati_.DtM;
    riccati.P.noalias() -= c_riccati_.KtDtM;
    riccati.P.noalias() -= c_riccati_.KtDtM.transpose();
    riccati.s.noalias() -= kkt_matrix.Phix().transpose() * riccati.m();
  }
}

void RiccatiFactorizer::backwardRiccatiRecursion(
    const SplitRiccatiFactorization &riccati_next, SplitKKTMatrix &kkt_matrix,
    SplitKKTResidual &kkt_residual, SplitRiccatiFactorization &riccati,
    LQRPolicy &lqr_policy, const bool sto, const bool has_next_sto_phase) {
  backwardRiccatiRecursion(riccati_next, kkt_matrix, kkt_residual, riccati,
                           lqr_policy);
  if (!sto) {
    riccati.Psi.setZero();
    riccati.xi = 0.;
    riccati.chi = 0.;
    riccati.eta = 0.;
    return;
  }

  backward_recursion_.factorizeHamiltonian(riccati_next, kkt_matrix, riccati,
                                           has_next_sto_phase);
  lqr_policy.W.setZero();
  if (kkt_matrix.dims() > 0) {
    lqr_policy.T.noalias() = -c_riccati_.Ginv * riccati.psi_u;
    lqr_policy.T.noalias() -=
        c_riccati_.SinvDGinv().transpose() * kkt_matrix.Phit();
    if (has_next_sto_phase) {
      lqr_policy.W.noalias() = -c_riccati_.Ginv * riccati.phi_u;
    }
    riccati.mt().noalias() = llt_s_.solve(kkt_matrix.Phit());
    riccati.mt().noalias() -= c_riccati_.SinvDGinv() * riccati.psi_u;
    if (has_next_sto_phase) {
      riccati.mt_next().noalias() = -c_riccati_.SinvDGinv() * riccati.phi_u;
    } else {
      riccati.mt_next().setZero();
    }
  } else {
    lqr_policy.T.noalias() = -llt_.solve(riccati.psi_u);
    if (has_next_sto_phase) {
      lqr_policy.W.noalias() = -llt_.solve(riccati.phi_u);
    }
  }
  backward_recursion_.factorizeSTOFactorization(riccati_next, kkt_matrix,
                                                kkt_residual, lqr_policy,
                                                riccati, has_next_sto_phase);
  if (kkt_matrix.dims() == 0)
    return;

  riccati.Psi.noalias() += riccati.M().transpose() * kkt_matrix.Phit();
  riccati.xi += riccati.mt().dot(kkt_matrix.Phit());
  if (has_next_sto_phase) {
    riccati.chi += riccati.mt_next().dot(kkt_matrix.Phit());
  }
  riccati.eta += riccati.m().dot(kkt_matrix.Phit());
}

void RiccatiFactorizer::backwardRiccatiRecursionPhaseTransition(
    const SplitRiccatiFactorization &riccati,
    SplitRiccatiFactorization &riccati_m, STOPolicy &sto_policy,
    const bool has_next_sto_phase) const {
  riccati_m.P = riccati.P;
  riccati_m.s = riccati.s;
  riccati_m.Psi.setZero();
  riccati_m.Phi = riccati.Psi;
  riccati_m.xi = 0.0;
  riccati_m.chi = 0.0;
  riccati_m.rho = riccati.xi;
  riccati_m.eta = 0.0;
  riccati_m.iota = riccati.eta;
  if (has_next_sto_phase) {
    double sgm = riccati.xi - 2.0 * riccati.chi + riccati.rho;
    if ((sgm * max_dts0_) < std::abs(riccati.eta - riccati.iota) ||
        sgm < eps_) {
      sgm = std::abs(sgm) + std::abs(riccati.eta - riccati.iota) / max_dts0_;
    }
    sto_policy.dtsdx = -(1.0 / sgm) * (riccati.Psi - riccati.Phi);
    sto_policy.dtsdts = (1.0 / sgm) * (riccati.xi - riccati.chi);
    sto_policy.dts0 = -(1.0 / sgm) * (riccati.eta - riccati.iota);
    riccati_m.s.noalias() += (1.0 / sgm) * (riccati.Psi - riccati.Phi) *
                             (riccati.eta - riccati.iota);
    riccati_m.Phi.noalias() -=
        (1.0 / sgm) * (riccati.Psi - riccati.Phi) * (riccati.xi - riccati.chi);
    riccati_m.rho = riccati.xi - (1.0 / sgm) * (riccati.xi - riccati.chi) *
                                     (riccati.xi - riccati.chi);
    riccati_m.iota = riccati.eta - (1.0 / sgm) * (riccati.xi - riccati.chi) *
                                       (riccati.eta - riccati.iota);
  }
}

void RiccatiFactorizer::backwardRiccatiRecursion(
    const SplitRiccatiFactorization &riccati_next, SplitKKTMatrix &kkt_matrix,
    SplitKKTResidual &kkt_residual, SplitRiccatiFactorization &riccati) {
  backward_recursion_.factorizeKKTMatrix(riccati_next, kkt_matrix);
  backward_recursion_.factorizeRiccatiFactorization(riccati_next, kkt_matrix,
                                                    kkt_residual, riccati);
}

void RiccatiFactorizer::backwardRiccatiRecursion(
    const SplitRiccatiFactorization &riccati_next, SplitKKTMatrix &kkt_matrix,
    SplitKKTResidual &kkt_residual, SplitRiccatiFactorization &riccati,
    const bool sto) {
  backwardRiccatiRecursion(riccati_next, kkt_matrix, kkt_residual, riccati);
  if (sto) {
    backward_recursion_.factorizeSTOFactorization(riccati_next, kkt_matrix,
                                                  kkt_residual, riccati);
  }
}

void forwardRiccatiRecursion(const SplitKKTMatrix &kkt_matrix,
                             const SplitKKTResidual &kkt_residual,
                             const LQRPolicy &lqr_policy, SplitDirection &d,
                             SplitDirection &d_next, const bool sto,
                             const bool has_next_sto_phase) {
  d.du.noalias() = lqr_policy.K * d.dx;
  d.du.noalias() += lqr_policy.k;
  if (sto) {
    d.du.noalias() += lqr_policy.T * (d.dts_next - d.dts);
    if (has_next_sto_phase) {
      d.du.noalias() -= lqr_policy.W * d.dts_next;
    }
  }
  d_next.dx = kkt_residual.Fx;
  d_next.dx.noalias() += kkt_matrix.Fxx * d.dx;
  d_next.dv().noalias() += kkt_matrix.Fvu * d.du;
  if (sto) {
    d_next.dx.noalias() += kkt_matrix.fx * (d.dts_next - d.dts);
  }
  d_next.dts = d.dts;
  d_next.dts_next = d.dts_next;
}

void forwardRiccatiRecursion(const SplitKKTMatrix &kkt_matrix,
                             const SplitKKTResidual &kkt_residual,
                             const SplitDirection &d, SplitDirection &d_next) {
  d_next.dx = kkt_residual.Fx;
  d_next.dx.noalias() += kkt_matrix.Fxx * d.dx;
  d_next.dts = d.dts;
  d_next.dts_next = d.dts_next;
}

void computeSwitchingTimeDirection(const STOPolicy &sto_policy,
                                   SplitDirection &d,
                                   const bool has_prev_sto_phase) {
  d.dts_next = sto_policy.dtsdx.dot(d.dx) + sto_policy.dts0;
  if (has_prev_sto_phase) {
    d.dts_next += sto_policy.dtsdts * d.dts;
  }
}

void computeCostateDirection(const SplitRiccatiFactorization &riccati,
                             SplitDirection &d, const bool sto,
                             const bool has_next_sto_phase) {
  d.dlmdgmm.noalias() = riccati.P * d.dx - riccati.s;
  if (sto) {
    d.dlmdgmm.noalias() += riccati.Psi * (d.dts_next - d.dts);
    if (has_next_sto_phase) {
      d.dlmdgmm.noalias() -= riccati.Phi * d.dts_next;
    }
  }
}

void computeCostateDirection(const SplitRiccatiFactorization &riccati,
                             SplitDirection &d, const bool sto) {
  d.dlmdgmm.noalias() = riccati.P * d.dx - riccati.s;
  if (sto) {
    d.dlmdgmm.noalias() -= riccati.Phi * d.dts_next;
  }
}

void computeLagrangeMultiplierDirection(
    const SplitRiccatiFactorization &riccati, SplitDirection &d, const bool sto,
    const bool has_next_sto_phase) {
  d.setSwitchingConstraintDimension(riccati.dims());
  d.dxi().noalias() = riccati.M() * d.dx;
  d.dxi().noalias() += riccati.m();
  if (sto) {
    d.dxi().noalias() += riccati.mt() * (d.dts_next - d.dts);
    if (has_next_sto_phase) {
      d.dxi().noalias() -= riccati.mt_next() * d.dts_next;
    }
  }
}

} // namespace robotoc