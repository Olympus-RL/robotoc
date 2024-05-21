#ifndef ROBOTOC_TRAJECTORY_REF_HPP_
#define ROBOTOC_TRAJECTORY_REF_HPP_

#include "robotoc/cost/configuration_space_ref_base.hpp"
#include "robotoc/ocp/time_discretization.hpp"
#include "robotoc/robot/robot.hpp"
#include "robotoc/utils/aligned_vector.hpp"

namespace robotoc {

class TrajectoryRef : public ConfigurationSpaceRefBase {
public:
  using StageTraj = aligned_vector<Eigen::VectorXd>;
  using Traj = std::vector<StageTraj>;
  TrajectoryRef(const Robot &robot, const Traj &knotPoints);
  ~TrajectoryRef() = default; // Default destructor

  void discretize(const Robot &robot,
                  const TimeDiscretization &time_discretization) override;
  void updateRef(const Robot &robot, const GridInfo &grid_info,
                 Eigen::VectorXd &q_ref) const override;
  bool isActive(const GridInfo &grid_info) const override;

private:
  void interpolate_phase(const Robot &robot, const int phase,
                         const int num_grids);

  Traj knotPoints_, refrence_;
  std::vector<int> num_knots_;
  int num_phases_, dimq_;
};

} // namespace robotoc

#endif // ROBOTOC_TRAJECTORY_REF_HPP_