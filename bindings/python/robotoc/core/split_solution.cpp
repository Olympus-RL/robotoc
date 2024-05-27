#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "robotoc/core/split_solution.hpp"
#include "robotoc/utils/pybind11_macros.hpp"

namespace robotoc {
namespace python {

namespace py = pybind11;

PYBIND11_MODULE(split_solution, m) {
  py::class_<SplitSolution>(m, "SplitSolution")
      .def(py::init<const Robot &>())
      .def(py::init<>())
      .def("set_contact_status",
           static_cast<void (SplitSolution::*)(const ContactStatus &)>(
               &SplitSolution::setContactStatus),
           py::arg("contact_status"))
      .def("set_contact_status",
           static_cast<void (SplitSolution::*)(const ImpactStatus &)>(
               &SplitSolution::setContactStatus),
           py::arg("contact_status"))
      .def("set_contact_status",
           static_cast<void (SplitSolution::*)(const SplitSolution &)>(
               &SplitSolution::setContactStatus),
           py::arg("other"))
      .def("set_switching_constraint_dimension",
           &SplitSolution::setSwitchingConstraintDimension, py::arg("dims"))
      .def("set_gf_stack", &SplitSolution::set_gf_stack)
      .def("set_gf_vector", &SplitSolution::set_gf_vector)
      .def("set_rhomu_stack", &SplitSolution::set_rhomu_stack)
      .def("set_rhomu_vector", &SplitSolution::set_rhomu_vector)
      .def("dimf", &SplitSolution::dimf)
      .def("dims", &SplitSolution::dims)
      .def("is_contact_active",
           static_cast<bool (SplitSolution::*)(const int) const>(
               &SplitSolution::isContactActive),
           py::arg("contact_index"))
      .def("is_contact_active",
           static_cast<std::vector<bool> (SplitSolution::*)() const>(
               &SplitSolution::isContactActive))
      .def_readwrite("q", &SplitSolution::q)
      .def_readwrite("v", &SplitSolution::v)
      .def_readwrite("u", &SplitSolution::u)
      .def_readwrite("a", &SplitSolution::a)
      .def_readwrite("dv", &SplitSolution::dv)
      .def_readwrite("f", &SplitSolution::f)
      .def_readwrite("lmd", &SplitSolution::lmd)
      .def_readwrite("gmm", &SplitSolution::gmm)
      .def_readwrite("beta", &SplitSolution::beta)
      .def_readwrite("mu", &SplitSolution::mu)
      .def_readwrite("nu_passive", &SplitSolution::nu_passive)
      .def_property(
          "gf_stack",
          static_cast<const Eigen::VectorBlock<const Eigen::VectorXd> (
              SplitSolution::*)() const>(&SplitSolution::gf_stack),
          static_cast<Eigen::VectorBlock<Eigen::VectorXd> (SplitSolution::*)()>(
              &SplitSolution::f_stack))
      .def_property(
          "rhomu_stack",
          static_cast<const Eigen::VectorBlock<const Eigen::VectorXd> (
              SplitSolution::*)() const>(&SplitSolution::rhomu_stack),
          static_cast<Eigen::VectorBlock<Eigen::VectorXd> (SplitSolution::*)()>(
              &SplitSolution::rhomu_stack))
      .def_property(
          "xi_stack",
          static_cast<const Eigen::VectorBlock<const Eigen::VectorXd> (
              SplitSolution::*)() const>(&SplitSolution::xi_stack),
          static_cast<Eigen::VectorBlock<Eigen::VectorXd> (SplitSolution::*)()>(
              &SplitSolution::xi_stack))
      .def("dimf", &SplitSolution::dimf)
      .def("dims", &SplitSolution::dims)
      .def("dimg", &SplitSolution::dimg)
          DEFINE_ROBOTOC_PYBIND11_CLASS_CLONE(SplitSolution)
              DEFINE_ROBOTOC_PYBIND11_CLASS_PRINT(SplitSolution);
}

} // namespace python
} // namespace robotoc