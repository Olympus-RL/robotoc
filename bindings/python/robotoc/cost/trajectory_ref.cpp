#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "robotoc/cost/trajectory_ref.hpp"

namespace robotoc {
namespace python {

namespace py = pybind11;

PYBIND11_MODULE(trajectory_ref, m) {
  py::class_<TrajectoryRef, ConfigurationSpaceRefBase,
             std::shared_ptr<TrajectoryRef>>(m, "TrajectoryRef")
      .def(py::init<const robotoc::Robot &,
                    const robotoc::TrajectoryRef::Traj &>(),
           py::arg("robot"), py::arg("knot_points"));
}

} // namespace python
} // namespace robotoc