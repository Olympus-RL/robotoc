#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "robotoc/core/split_direction.hpp"
#include "robotoc/utils/pybind11_macros.hpp"

namespace robotoc {
namespace python {

namespace py = pybind11;

PYBIND11_MODULE(split_direction, m) {
  py::class_<SplitDirection>(m, "SplitDirection")
      .def(py::init<const Robot &>())
      .def(py::init<>())
      .def("set_contact_dimension", &SplitDirection::setContactDimension,
           py::arg("dimf"))
      .def("set_switching_constraint_dimension",
           &SplitDirection::setSwitchingConstraintDimension, py::arg("dims"))
      .def("dimf", &SplitDirection::dimf)
      .def("dims", &SplitDirection::dims)
      .def_readwrite("dx", &SplitDirection::dx)
      .def_readwrite("du", &SplitDirection::du)
      .def_readwrite("dlmdgmm", &SplitDirection::dlmdgmm)
      .def_readwrite("dnu_passive", &SplitDirection::dnu_passive)
      .def_property(
          "dq",
          static_cast<const Eigen::VectorBlock<const Eigen::VectorXd> (
              SplitDirection::*)() const>(&SplitDirection::dq),
          static_cast<Eigen::VectorBlock<Eigen::VectorXd> (
              SplitDirection::*)()>(&SplitDirection::dq))
      .def_property(
          "dv",
          static_cast<const Eigen::VectorBlock<const Eigen::VectorXd> (
              SplitDirection::*)() const>(&SplitDirection::dv),
          static_cast<Eigen::VectorBlock<Eigen::VectorXd> (
              SplitDirection::*)()>(&SplitDirection::dv))
      .def_property(
          "dagf",
          static_cast<const Eigen::VectorBlock<const Eigen::VectorXd> (
              SplitDirection::*)() const>(&SplitDirection::dagf),
          static_cast<Eigen::VectorBlock<Eigen::VectorXd> (
              SplitDirection::*)()>(&SplitDirection::dagf))
      .def_property(
          "da",
          static_cast<const Eigen::VectorBlock<const Eigen::VectorXd> (
              SplitDirection::*)() const>(&SplitDirection::da),
          static_cast<Eigen::VectorBlock<Eigen::VectorXd> (
              SplitDirection::*)()>(&SplitDirection::da))
      .def_property(
          "ddv",
          static_cast<const Eigen::VectorBlock<const Eigen::VectorXd> (
              SplitDirection::*)() const>(&SplitDirection::ddv),
          static_cast<Eigen::VectorBlock<Eigen::VectorXd> (
              SplitDirection::*)()>(&SplitDirection::ddv))
      .def_property(
          "dgf",
          static_cast<const Eigen::VectorBlock<const Eigen::VectorXd> (
              SplitDirection::*)() const>(&SplitDirection::dgf),
          static_cast<Eigen::VectorBlock<Eigen::VectorXd> (
              SplitDirection::*)()>(&SplitDirection::dgf))
      .def_property(
          "dlmd",
          static_cast<const Eigen::VectorBlock<const Eigen::VectorXd> (
              SplitDirection::*)() const>(&SplitDirection::dlmd),
          static_cast<Eigen::VectorBlock<Eigen::VectorXd> (
              SplitDirection::*)()>(&SplitDirection::dlmd))
      .def_property(
          "dgmm",
          static_cast<const Eigen::VectorBlock<const Eigen::VectorXd> (
              SplitDirection::*)() const>(&SplitDirection::dgmm),
          static_cast<Eigen::VectorBlock<Eigen::VectorXd> (
              SplitDirection::*)()>(&SplitDirection::dgmm))
      .def_property(
          "dbetarhomu",
          static_cast<const Eigen::VectorBlock<const Eigen::VectorXd> (
              SplitDirection::*)() const>(&SplitDirection::dbetarhomu),
          static_cast<Eigen::VectorBlock<Eigen::VectorXd> (
              SplitDirection::*)()>(&SplitDirection::dbetarhomu))
      .def_property(
          "dbeta",
          static_cast<const Eigen::VectorBlock<const Eigen::VectorXd> (
              SplitDirection::*)() const>(&SplitDirection::dbeta),
          static_cast<Eigen::VectorBlock<Eigen::VectorXd> (
              SplitDirection::*)()>(&SplitDirection::dbeta))
      .def_property(
          "drhomu",
          static_cast<const Eigen::VectorBlock<const Eigen::VectorXd> (
              SplitDirection::*)() const>(&SplitDirection::drhomu),
          static_cast<Eigen::VectorBlock<Eigen::VectorXd> (
              SplitDirection::*)()>(&SplitDirection::drhomu))
      .def_property(
          "dxi",
          static_cast<const Eigen::VectorBlock<const Eigen::VectorXd> (
              SplitDirection::*)() const>(&SplitDirection::dxi),
          static_cast<Eigen::VectorBlock<Eigen::VectorXd> (
              SplitDirection::*)()>(&SplitDirection::dxi))
      .def("dimf", &SplitDirection::dimf)
      .def("dims", &SplitDirection::dims)
          DEFINE_ROBOTOC_PYBIND11_CLASS_CLONE(SplitDirection)
              DEFINE_ROBOTOC_PYBIND11_CLASS_PRINT(SplitDirection);
}

} // namespace python
} // namespace robotoc