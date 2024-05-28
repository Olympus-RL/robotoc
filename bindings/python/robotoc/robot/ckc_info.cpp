#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "robotoc/robot/ckc_info.hpp"
#include "robotoc/utils/pybind11_macros.hpp"

#include <iostream>

namespace robotoc {
namespace python {

namespace py = pybind11;

PYBIND11_MODULE(ckc_info, m) {
  py::class_<CKCInfo>(m, "CKCInfo")
      .def(py::init<const std::string &, const std::string &, const double>(),
           py::arg("frame_0"), py::arg("frame_1"),
           py::arg("baumgarte_time_step"))
      .def(py::init<const std::string &, const std::string &, const double,
                    const double>(),
           py::arg("frame_0"), py::arg("frame_0"),
           py::arg("baumgarte_position_gain"),
           py::arg("baumgarte_velocity_gain"))
      .def(py::init<>())
      .def_readwrite("frame_0", &CKCInfo::frame_0)
      .def_readwrite("frame_1", &CKCInfo::frame_1)
      .def_readwrite("baumgarte_position_gain",
                     &CKCInfo::baumgarte_position_gain)
      .def_readwrite("baumgarte_velocity_gain",
                     &CKCInfo::baumgarte_velocity_gain)
          DEFINE_ROBOTOC_PYBIND11_CLASS_CLONE(CKCInfo);
}

} // namespace python
} // namespace robotoc