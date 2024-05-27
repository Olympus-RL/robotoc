#include "robotoc/core/split_direction.hpp"

#include <random>

namespace robotoc {

SplitDirection::SplitDirection(const Robot &robot)
    : dx(Eigen::VectorXd::Zero(2 * robot.dimv())),
      du(Eigen::VectorXd::Zero(robot.dimu())),
      dlmdgmm(Eigen::VectorXd::Zero(2 * robot.dimv())),
      dnu_passive(Eigen::VectorXd::Zero(robot.dim_passive())), dts(0.0),
      dts_next(0.0), dagf_full_(Eigen::VectorXd::Zero(
                         robot.dimv() + robot.dimg() + robot.max_dimf())),
      dbetarhomu_full_(Eigen::VectorXd::Zero(robot.dimv() + robot.dimg() +
                                             robot.max_dimf())),
      dxi_full_(Eigen::VectorXd::Zero(robot.max_dimf())), dimv_(robot.dimv()),
      dimu_(robot.dimu()), dim_passive_(robot.dim_passive()), dimf_(0),
      dimg_(robot.dimg()), dims_(0) {}

SplitDirection::SplitDirection()
    : dx(), du(), dlmdgmm(), dnu_passive(), dts(0.0), dts_next(0.0),
      dagf_full_(), dbetarhomu_full_(), dxi_full_(), dimv_(0), dimu_(0),
      dim_passive_(0), dimf_(0), dims_(0) {}

bool SplitDirection::isDimensionConsistent() const {
  if (dx.size() != 2 * dimv_)
    return false;
  if (du.size() != dimu_)
    return false;
  if (dlmdgmm.size() != 2 * dimv_)
    return false;
  if (dnu_passive.size() != dim_passive_)
    return false;
  return true;
}

bool SplitDirection::isApprox(const SplitDirection &other) const {
  assert(isDimensionConsistent());
  assert(other.isDimensionConsistent());
  assert(dimf() == other.dimf());
  assert(dims() == other.dims());
  if (!dx.isApprox(other.dx)) {
    return false;
  }
  if (!du.isApprox(other.du)) {
    return false;
  }
  if (!dagf().isApprox(other.dagf())) {
    return false;
  }
  if (!dlmdgmm.isApprox(other.dlmdgmm)) {
    return false;
  }
  if (!dbetarhomu().isApprox(other.dbetarhomu())) {
    return false;
  }
  if (!dxi().isApprox(other.dxi())) {
    return false;
  }
  if (!dnu_passive.isApprox(other.dnu_passive)) {
    return false;
  }
  Eigen::VectorXd vec(2), other_vec(2);
  vec << dts, dts_next;
  other_vec << other.dts, other.dts_next;
  if (!vec.isApprox(other_vec)) {
    return false;
  }
  return true;
}

void SplitDirection::setRandom() {
  assert(isDimensionConsistent());
  dx.setRandom();
  du.setRandom();
  dagf().setRandom();
  dlmdgmm.setRandom();
  dbetarhomu().setRandom();
  dnu_passive.setRandom();
  dxi().setRandom();
  dts = Eigen::VectorXd::Random(1)[0];
  dts_next = Eigen::VectorXd::Random(1)[0];
}

void SplitDirection::setRandom(const ContactStatus &contact_status) {
  setContactDimension(contact_status.dimf());
  setRandom();
}

void SplitDirection::setRandom(const ImpactStatus &impact_status) {
  setContactDimension(impact_status.dimf());
  setRandom();
}

void SplitDirection::setRandom(const ContactStatus &contact_status,
                               const ImpactStatus &impact_status) {
  setContactDimension(contact_status.dimf());
  setSwitchingConstraintDimension(impact_status.dimf());
  setRandom();
}

SplitDirection SplitDirection::Random(const Robot &robot) {
  SplitDirection d(robot);
  d.setRandom();
  return d;
}

SplitDirection SplitDirection::Random(const Robot &robot,
                                      const ContactStatus &contact_status) {
  SplitDirection d(robot);
  d.setRandom(contact_status);
  return d;
}

SplitDirection SplitDirection::Random(const Robot &robot,
                                      const ImpactStatus &impact_status) {
  SplitDirection d(robot);
  d.setRandom(impact_status);
  return d;
}

SplitDirection SplitDirection::Random(const Robot &robot,
                                      const ContactStatus &contact_status,
                                      const ImpactStatus &impact_status) {
  SplitDirection d(robot);
  d.setRandom(contact_status, impact_status);
  return d;
}

void SplitDirection::disp(std::ostream &os) const {
  os << "SplitDirection:"
     << "\n";
  os << "  dq       = " << dq().transpose() << "\n";
  os << "  dv       = " << dv().transpose() << "\n";
  os << "  du       = " << du.transpose() << "\n";
  os << "  da (ddv) = " << da().transpose() << "\n";
  if (dimf() > 0) {
    os << "  dgf       = " << dgf().transpose() << "\n";
  }
  os << "  dlmd     = " << dlmd().transpose() << "\n";
  os << "  dgmm     = " << dgmm().transpose() << "\n";
  os << "  dbeta    = " << dbeta().transpose() << "\n";
  if (dimf() + dimg() > 0) {
    os << "  drhomu      = " << drhomu().transpose() << "\n";
  }
  if (dims() > 0) {
    os << "  dxi      = " << dxi().transpose() << std::flush;
  }
}

std::ostream &operator<<(std::ostream &os, const SplitDirection &d) {
  d.disp(os);
  return os;
}

} // namespace robotoc