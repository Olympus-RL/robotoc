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
  using Matrix64d = Eigen::Matrix<double, 6, 4>;
  using Matrix34d = Eigen::Matrix<double, 3, 4>;
  using Matrix24d = Eigen::Matrix<double, 2, 4>;

  ///
  /// @brief Construct a point contact model.
  /// @param[in] model The pinocchio model. Before calling this constructor,
  /// pinocchio model must be initialized, e.g., by pinocchio::buildModel().
  /// @param[in] contact_model_info Info of the point contact model.
  ///
  CKC(const ::pinocchio::Model &model, const CKCInfo &contact_model_info);

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

  template <typename ConfigVectorType, typename TangentVectorType1,
            typename TangentVectorType2>
  void updateKinematics(const Eigen::MatrixBase<ConfigVectorType> &q,
                        const Eigen::MatrixBase<TangentVectorType1> &v,
                        const Eigen::MatrixBase<TangentVectorType2> &a);
  ///
  /// @brief Converts the 3D constraint forces in world coordinate to the
  /// corresponding joint spatial forces.
  /// @param[in] constraint_force The 2D constraint forces in the the common
  /// ancestor frame frame.
  /// @param[out] Q The generalized constraint force
  ///

  void computeJointForceFromConstraintForce(
      const Eigen::Vector2d &contact_force,
      pinocchio::container::aligned_vector<pinocchio::Force> &joint_forces);

  void computeGeneralizedForceFromConstraintForce(
      const Eigen::Vector2d &constraint_force, Eigen::VectorXd &Q);

  template <typename MatrixType1>
  void
  computeConstrsaintForceDerivative(const Eigen::Vector2d &g,
                                    const Eigen::MatrixBase<MatrixType1> &dQdq);

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
      const Eigen::MatrixBase<VectorType1> &baumgarte_residual);

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
  ;

  ///
  /// @brief Gets the contact model info.
  /// @return const reference to the contact model info.
  ///
  const CKCInfo &ckcInfo() const;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  CKCInfo info_;

  int dimq_, dimv_, start_q_idx_, start_v_idx_, frame_0_idx_, frame_1_idx_,
      joint_0_idx_, joint_1_idx_;
  // classical velocity of frame
  ::pinocchio::Model submodel_;
  ::pinocchio::Data subdata_;
  ::pinocchio::Motion v_frame_, a_frame_;
  Eigen::Matrix3d v_linear_skew_, v_angular_skew_, alpha_skew_, r_skew_;
  Matrix64d J_frame_, J_frame_dot_, frame_v_partial_dq_, frame_a_partial_dq_,
      frame_a_partial_dv_, frame_a_partial_da_;
  Matrix24d Jc_;
  Matrix34d vel_partial_dq_;
  robotoc::SE3 jXf_0_, jXf_1_;
};

} // namespace robotoc

#include "robotoc/robot/ckc.hxx"

#endif // ROBOTOC_CKC_HPP_