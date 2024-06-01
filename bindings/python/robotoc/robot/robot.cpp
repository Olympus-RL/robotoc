#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "robotoc/robot/robot.hpp"
#include "robotoc/utils/pybind11_macros.hpp"

#include <iostream>

namespace robotoc {
namespace python {

namespace py = pybind11;

PYBIND11_MODULE(robot, m) {
  py::class_<Robot>(m, "Robot")
      .def(py::init<const RobotModelInfo &>(), py::arg("robot_model_info"))
      .def(py::init<>())
      .def("get_ckc_residual",
           [](Robot &self) {
             Eigen::VectorXd ckc_residual =
                 Eigen::VectorXd::Zero(self.dimf_ckc());
             self.computeCKCResidual(ckc_residual);
             return ckc_residual;
           })
      .def("get_ckc_jacobian",
           [](Robot &self) {
             Eigen::MatrixXd ckc_jacobian =
                 Eigen::MatrixXd::Zero(self.dimf_ckc(), self.dimv());
             self.computeCKCJacobian(ckc_jacobian);
             return ckc_jacobian;
           })
      .def("get_contact_position_residual",
           [](Robot &self, robotoc::ContactStatus &contact_status) {
             Eigen::VectorXd contact_residual =
                 Eigen::VectorXd::Zero(contact_status.dimf());
             self.computeContactPositionResidual(contact_status,
                                                 contact_residual);
             return contact_residual;
           })
      .def("get_contact_position_jacobian",
           [](Robot &self, robotoc::ContactStatus &contact_status) {
             Eigen::MatrixXd contact_jacobian =
                 Eigen::MatrixXd::Zero(contact_status.dimf(), self.dimv());
             self.computeContactPositionDerivative(contact_status,
                                                   contact_jacobian);
             return contact_jacobian;
           })
      .def(
          "rnea",
          [](Robot &self, const Eigen::VectorXd &q, const Eigen::VectorXd &v,
             const Eigen::VectorXd &a) {
            Eigen::VectorXd tau = Eigen::VectorXd::Zero(self.dimv());
            self.RNEA(q, v, a, tau);
            return tau;
          },
          py::arg("q"), py::arg("v"), py::arg("a"))
      .def(
          "integrate_configuration",
          [](const Robot &self, const Eigen::VectorXd &q,
             const Eigen::VectorXd &v, const double dt) {
            Eigen::VectorXd qout = Eigen::VectorXd::Zero(self.dimq());
            self.integrateConfiguration(q, v, dt, qout);
            return qout;
          },
          py::arg("q"), py::arg("v"), py::arg("dt"))
      .def(
          "subtract_configuration",
          [](const Robot &self, const Eigen::VectorXd &qf,
             const Eigen::VectorXd &q0) {
            Eigen::VectorXd qdiff = Eigen::VectorXd::Zero(self.dimv());
            self.subtractConfiguration(qf, q0, qdiff);
            return qdiff;
          },
          py::arg("qf"), py::arg("q0"))
      .def(
          "interpolate_configuration",
          [](const Robot &self, const Eigen::VectorXd &q1,
             const Eigen::VectorXd &q2, const double t) {
            Eigen::VectorXd qout = Eigen::VectorXd::Zero(self.dimq());
            self.interpolateConfiguration(q1, q2, t, qout);
            return qout;
          },
          py::arg("q1"), py::arg("q2"), py::arg("t"))
      .def(
          "integrate_coeff_wise_jacobian",
          [](const Robot &self, const Eigen::VectorXd &q) {
            Eigen::MatrixXd J = Eigen::MatrixXd::Zero(self.dimq(), self.dimv());
            self.integrateCoeffWiseJacobian(q, J);
            return J;
          },
          py::arg("q"))

      .def(
          "get_frame_world_jacobian",
          [](Robot &self, const std::string &frame) {
            Eigen::MatrixXd J = Eigen::MatrixXd::Zero(6, self.dimv());
            self.getFrameWorldJacobian(frame, J);
            return J;
          },
          py::arg("frame"))
      .def(
          "update_kinematics",
          [](Robot &self, const Eigen::VectorXd &q) {
            self.updateKinematics(q);
          },
          py::arg("q"))
      .def(
          "update_kinematics",
          [](Robot &self, const Eigen::VectorXd &q, const Eigen::VectorXd &v) {
            self.updateKinematics(q, v);
          },
          py::arg("q"), py::arg("v"))
      .def(
          "update_kinematics",
          [](Robot &self, const Eigen::VectorXd &q, const Eigen::VectorXd &v,
             const Eigen::VectorXd &a) { self.updateKinematics(q, v, a); },
          py::arg("q"), py::arg("v"), py::arg("a"))
      .def(
          "forward_kinematics",
          [](Robot &self, const Eigen::VectorXd &q, bool compute_jacobians) {
            self.updateFrameKinematics(q, compute_jacobians);
          },
          py::arg("q"), py::arg("compute_jacobians") = false)
      .def(
          "forward_kinematics",
          [](Robot &self, const Eigen::VectorXd &q, const Eigen::VectorXd &v,
             bool compute_jacobians) {
            self.updateFrameKinematics(q, v, compute_jacobians);
          },
          py::arg("q"), py::arg("v"), py::arg("compute_jacobians") = false)
      .def(
          "forward_kinematics",
          [](Robot &self, const Eigen::VectorXd &q, const Eigen::VectorXd &v,
             const Eigen::VectorXd &a, bool compute_jacobians) {
            self.updateFrameKinematics(q, v, a, compute_jacobians);
          },
          py::arg("q"), py::arg("v"), py::arg("a"),
          py::arg("compute_jacobians") = false)
      .def("frame_position",
           static_cast<const Eigen::Vector3d &(Robot::*)(const int) const>(
               &Robot::framePosition),
           py::arg("frame_id"))
      .def("frame_position",
           static_cast<const Eigen::Vector3d &(Robot::*)(const std::string &)
                           const>(&Robot::framePosition),
           py::arg("frame_name"))
      .def("frame_rotation",
           static_cast<const Eigen::Matrix3d &(Robot::*)(const int) const>(
               &Robot::frameRotation),
           py::arg("frame_id"))
      .def("frame_rotation",
           static_cast<const Eigen::Matrix3d &(Robot::*)(const std::string &)
                           const>(&Robot::frameRotation),
           py::arg("frame_name"))
      .def("frame_placement",
           static_cast<const SE3 &(Robot::*)(const int) const>(
               &Robot::framePlacement),
           py::arg("frame_id"))
      .def("frame_placement",
           static_cast<const SE3 &(Robot::*)(const std::string &) const>(
               &Robot::framePlacement),
           py::arg("frame_name"))
      .def("com", &Robot::CoM)
      .def(
          "frame_linear_velocity",
          [](const Robot &self, const int frame_id) {
            return self.frameLinearVelocity(frame_id);
          },
          py::arg("frame_id"))
      .def(
          "frame_linear_velocity",
          [](const Robot &self, const std::string &frame_name) {
            return self.frameLinearVelocity(frame_name);
          },
          py::arg("frame_name"))
      .def(
          "frame_angular_velocity",
          [](const Robot &self, const int frame_id) {
            return self.frameAngularVelocity(frame_id);
          },
          py::arg("frame_id"))
      .def(
          "frame_angular_velocity",
          [](const Robot &self, const std::string &frame_name) {
            return self.frameAngularVelocity(frame_name);
          },
          py::arg("frame_name"))
      .def(
          "frame_spatial_velocity",
          [](const Robot &self, const int frame_id) {
            return self.frameSpatialVelocity(frame_id);
          },
          py::arg("frame_id"))
      .def(
          "frame_spatial_velocity",
          [](const Robot &self, const std::string &frame_name) {
            return self.frameSpatialVelocity(frame_name);
          },
          py::arg("frame_name"))
      .def("com_velocity", &Robot::CoMVelocity)
      .def(
          "transform_from_local_to_world",
          [](const Robot &self, const int frame_id,
             const Eigen::Vector3d &vec_local) {
            Eigen::Vector3d vec_world = Eigen::Vector3d::Zero();
            self.transformFromLocalToWorld(frame_id, vec_local, vec_world);
            return vec_world;
          },
          py::arg("frame_id"), py::arg("vec_local"))
      .def("generate_feasible_configuration",
           &Robot::generateFeasibleConfiguration)
      .def(
          "normalize_configuration",
          [](const Robot &self, Eigen::VectorXd &q) {
            self.normalizeConfiguration(q);
          },
          py::arg("q"))
      .def("S", &Robot::S)
      .def("Sbar", &Robot::Sbar)
      .def("create_contact_status", &Robot::createContactStatus)
      .def("create_impact_status", &Robot::createImpactStatus)
      .def("frame_id", &Robot::frameId, py::arg("frame_name"))
      .def("frame_name", &Robot::frameName, py::arg("frame_id"))
      .def("total_mass", &Robot::totalMass)
      .def("total_weight", &Robot::totalWeight)
      .def("dimq", &Robot::dimq)
      .def("dimv", &Robot::dimv)
      .def("dimu", &Robot::dimu)
      .def("max_dimf", &Robot::max_dimf)
      .def("dimf_ckc", &Robot::dimf_ckc)
      .def("dim_passive", &Robot::dim_passive)
      .def("has_floating_base", &Robot::hasFloatingBase)
      .def("numCKCs", &Robot::numCKCs)
      .def("max_num_contacts", &Robot::maxNumContacts)
      .def("max_num_point_contacts", &Robot::maxNumPointContacts)
      .def("max_num_surface_contacts", &Robot::maxNumSurfaceContacts)
      .def("contact_type", &Robot::contactType, py::arg("contact_index"))
      .def("contact_types", &Robot::contactTypes)
      .def("contact_frames", &Robot::contactFrames)
      .def("contact_frame_names", &Robot::contactFrameNames)
      .def("point_contact_frames", &Robot::pointContactFrames)
      .def("point_contact_frame_names", &Robot::pointContactFrameNames)
      .def("surface_contact_frames", &Robot::surfaceContactFrames)
      .def("surface_contact_frame_names", &Robot::surfaceContactFrameNames)
      .def("joint_effort_limit", &Robot::jointEffortLimit)
      .def("joint_velocity_limit", &Robot::jointVelocityLimit)
      .def("lower_joint_position_limit", &Robot::lowerJointPositionLimit)
      .def("upper_joint_position_limit", &Robot::upperJointPositionLimit)
      .def("set_joint_effort_limit", &Robot::setJointEffortLimit,
           py::arg("joint_effort_limit"))
      .def("set_joint_velocity_limit", &Robot::setJointVelocityLimit,
           py::arg("joint_velocity_limit"))
      .def("set_lower_joint_position_limit", &Robot::setLowerJointPositionLimit,
           py::arg("lower_joint_position_limit"))
      .def("set_upper_joint_position_limit", &Robot::setUpperJointPositionLimit,
           py::arg("upper_joint_position_limit"))
      .def("set_gravity", &Robot::setGravity, py::arg("gravity"))
      .def("robot_model_info", &Robot::robotModelInfo)
      .def("robot_properties", &Robot::robotProperties)
      .def("set_robot_properties", &Robot::setRobotProperties,
           py::arg("properties")) DEFINE_ROBOTOC_PYBIND11_CLASS_CLONE(Robot)
          DEFINE_ROBOTOC_PYBIND11_CLASS_PRINT(Robot);
}

} // namespace python
} // namespace robotoc