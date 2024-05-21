#include "robotoc/cost/trajectory_ref.hpp"
#include <cmath>

namespace robotoc {

TrajectoryRef::TrajectoryRef(const Robot &robot, const Traj &knotPoints)
    : knotPoints_(knotPoints), num_phases_(knotPoints.size()),
      refrence_(knotPoints.size(), StageTraj()), dimq_(robot.dimq()) {
  for (int i = 0; i < knotPoints.size(); ++i) {
    num_knots_.push_back(knotPoints[i].size());
  }
}

void TrajectoryRef::updateRef(const Robot &robot, const GridInfo &grid_info,
                              Eigen::VectorXd &q_ref) const {
  assert(isActive(grid_info));
  assert(q_ref.size() == dimq_);
  assert(grid_info.phase < num_phases_);
  assert(grid_info.stage_in_phase < refrence_[grid_info.phase].size());
  if (grid_info.type != GridType::Terminal) {
    q_ref = refrence_[grid_info.phase][grid_info.stage_in_phase];
  } else {
    q_ref = refrence_.back().back();
  }
}

void TrajectoryRef::discretize(const Robot &robot,
                               const TimeDiscretization &time_discretization) {
  int num_grids = 0;
  int last_phase = 0;
  for (int i = 0; i < time_discretization.size(); ++i) {
    if (isActive(time_discretization[i])) {
      if (last_phase == time_discretization[i].phase) {
        num_grids++;
      } else {
        interpolate_phase(robot, last_phase, num_grids);
        num_grids = 1;
        last_phase = time_discretization[i].phase;
      }
    }
  }
  interpolate_phase(robot, last_phase, num_grids);
}

bool TrajectoryRef::isActive(const GridInfo &grid_info) const {
  return grid_info.type != GridType::Impact;
}

void TrajectoryRef::interpolate_phase(const Robot &robot, int phase,
                                      int num_grids) {
  assert(phase < num_phases_);
  assert(phase >= 0);
  assert(num_grids > 0);

  refrence_[phase].resize(num_grids, Eigen::VectorXd::Zero(robot.dimq()));
  for (int i = 0; i < num_grids; ++i) {
    const double t = static_cast<double>(i) / num_grids * num_knots_[phase];

    const int t_minus = std::floor(t);
    const int t_plus = std::ceil(t);
    double alpha = t - t_minus;
    robot.interpolateConfiguration(knotPoints_[phase][t_minus],
                                   knotPoints_[phase][t_plus], alpha,
                                   refrence_[phase][i]);
  }
}

} // namespace robotoc