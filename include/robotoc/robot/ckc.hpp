#ifndef ROBOTOC_CKC_HPP_
#define ROBOTOC_CKC_HPP_

#include "Eigen/Core"
#include "pinocchio/container/aligned-vector.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/model.hpp"
#include "pinocchio/spatial/force.hpp"

#include "robotoc/robot/ckc_info.hpp"
#include "robotoc/robot/se3.hpp"

#include <iostream>

namespace robotoc {

///
/// @class CKC
/// @brief Closed Kinematic Chain (CKC) constraint.
class CKC {
public:
  using Matrix6xd = Eigen::Matrix<double, 6, Eigen::Dynamic>;
  using Matrix3xd = Eigen::Matrix<double, 3, Eigen::Dynamic>;

  ///
  /// @brief Construct a point contact model.
  /// @param[in] model The pinocchio model. Before calling this constructor,
  /// pinocchio model must be initialized, e.g., by pinocchio::buildModel().
  /// @param[in] contact_model_info Info of the point contact model.
  ///
  CKC(const pinocchio::Model &model, const CKCInfo &contact_model_info);

  ///
  /// @brief Default constructor.
  ///
  CKC();

  ///
  /// @brief Default destructor.
  ///
  ~CKC() = default;

  ///
  /// @brief Default copy constructor.
  ///
  CKC(const CKC &) = default;

  ///
  /// @brief Default copy assign operator.
  ///
  CKC &operator=(const CKC &) = default;

  ///
  /// @brief Default move constructor.
  ///
  CKC(CKC &&) noexcept = default;

  ///
  /// @brief Default move assign operator.
  ///
  CKC &operator=(CKC &&) noexcept = default;

  void kuk() const;

  ///
  /// @brief Converts the 3D constraint forces in world coordinate to the
  /// corresponding joint spatial forces.
  /// @param[in] contact_force The 3D constraint forces in the local frame.
  /// @param[out] joint_forces: The corresponding joint spatial forces.
  ///
  void computeJointForceFromConstraintForce(
      const Eigen::Matrix3d &R_0, const Eigen::Matrix3d &R_1,
      const Eigen::Vector3d &constraint_force,
      pinocchio::container::aligned_vector<pinocchio::Force> &joint_forces)
      const;

  ///
  /// @brief Computes the residual of the contact constraints considered by the
  /// Baumgarte's stabilization method. Before calling this function, kinematics
  /// of the robot model (frame position, velocity, and acceleration) must be
  /// updated.
  /// @param[in] model Pinocchio model of the robot.
  /// @param[in] data Pinocchio data of the robot kinematics.
  /// @param[out] baumgarte_residual Residual of the Bamgarte's constraint.
  /// Size must be 3.
  ///
  template <typename VectorType1>
  void computeBaumgarteResidual(
      const pinocchio::Model &model, const pinocchio::Data &data,
      const Eigen::MatrixBase<VectorType1> &baumgarte_residual) const;

  ///
  /// @brief Computes the partial derivatives of the contact constraints
  /// considered by the Baumgarte's stabilization method. Before calling this
  /// function, kinematics derivatives of the robot model (position, velocity,
  /// and acceleration) must be updated.
  /// @param[in] model Pinocchio model of the robot.
  /// @param[in] data Pinocchio data of the robot kinematics.
  /// @param[out] baumgarte_partial_dq The partial derivative with respect to
  /// the configuaration. Size must be 3 x Robot::dimv().
  /// @param[out] baumgarte_partial_dv The partial derivative with respect to
  /// the velocity. Size must be 3 x Robot::dimv().
  /// @param[out] baumgarte_partial_da The partial derivative  with respect to
  /// the acceleration. Size must be 3 x Robot::dimv().
  ///
  template <typename MatrixType1, typename MatrixType2, typename MatrixType3>
  void computeBaumgarteDerivatives(
      const pinocchio::Model &model, pinocchio::Data &data,
      const Eigen::MatrixBase<MatrixType1> &baumgarte_partial_dq,
      const Eigen::MatrixBase<MatrixType2> &baumgarte_partial_dv,
      const Eigen::MatrixBase<MatrixType3> &baumgarte_partial_da);

  ///
  /// @brief Sets the gain parameters of the Baumgarte's stabilization method.
  /// @param[in] baumgarte_position_gain The position gain of the Baumgarte's
  /// stabilization method. Must be non-negative.
  /// @param[in] baumgarte_velocity_gain The velocity gain of the Baumgarte's
  /// stabilization method. Must be non-negative.
  ///
  void setBaumgarteGains(const double baumgarte_position_gain,
                         const double baumgarte_velocity_gain);

  ///
  /// @brief Computes the residual of the contact position constraints. Before
  /// calling this function, kinematics of the robot model (frame position) must
  /// be updated.
  /// @param[in] model Pinocchio model of the robot.
  /// @param[in] data Pinocchio data of the robot kinematics.
  /// @param[in] desired_contact_position Desired contact position. Size must
  /// be 3.
  /// @param[out] position_residual Residual of the contact constraint. Size
  /// must be 3.
  ///
  template <typename VectorType1>
  void computeCKCResidual(
      const pinocchio::Model &model, const pinocchio::Data &data,
      const Eigen::MatrixBase<VectorType1> &position_residual) const;

  ///
  /// @brief Gets the contact model info.
  /// @return const reference to the contact model info.
  ///
  const CKCInfo &ckcInfo() const;
  const int &frame_0_idx() const;
  const int &frame_1_idx() const;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  CKCInfo info_;
  int frame_0_idx_, parent_0_joint_idx_, dimv_;
  int frame_1_idx_, parent_1_joint_idx_;
  SE3 jXf_0_, jXf_1_;
  // classical velocity of frame
  pinocchio::Motion v_frame_, a_frame_;
  Eigen::Matrix3d v_linear_skew_, v_angular_skew_, r_skew_, alpha_skew_;
  Matrix6xd J_frame_, J_frame_dot_, frame_v_partial_dq_, frame_a_partial_dq_,
      frame_a_partial_dv_, frame_a_partial_da_;
  Matrix3xd vel_partial_dq_;
};

} // namespace robotoc

#include "robotoc/robot/ckc.hxx"

#endif // ROBOTOC_CKC_HPP_