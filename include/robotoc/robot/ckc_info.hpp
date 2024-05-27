#ifndef ROBOTOC_CKC_INFO_HPP_
#define ROBOTOC_CKC_INFO_HPP_

#include <string>
#include <Eigen/Core>


using namespace Eigen;

namespace robotoc {

///
/// @class CKCInfo
/// @brief Info of a 5-bar linkage closed kinematic chain. 
///
struct CKCInfo {
  ///
  /// @brief Construct a contact model info.
  /// @param[in] frame_0 Name of the first contact frame.
  /// @param[in] frame_1 Name of the second contact frame.
  /// @param[in] baumgarte_time_step Time step parameter of the Baumgarte's 
  /// stabilization method. Must be positive.
  ///
  CKCInfo(const std::string& frame_0, const std::string& frame_1, const double baumgarte_time_step);

  ///
  /// @brief Construct a contact model info.
  /// @param[in] frame Name of the contact frame.
  /// @param[in] baumgarte_position_gain The position gain of the Baumgarte's 
  /// stabilization method. Must be non-negative.
  /// @param[in] baumgarte_velocity_gain The velocity gain of the Baumgarte's 
  /// stabilization method. Must be non-negative.
  ///
  CKCInfo(const std::string& frame_0, 
          const std::string& frame_1, 
          const double baumgarte_position_gain,
          const double baumgarte_velocity_gain);

  ///
  /// @brief Default constructor. 
  ///
  CKCInfo() = default;

  ///
  /// @brief Default destructor. 
  ///
  ~CKCInfo() = default;

  ///
  /// @brief Default copy constructor. 
  ///
  CKCInfo(const CKCInfo&) = default;

  ///
  /// @brief Default copy assign operator. 
  ///
  CKCInfo& operator=(const CKCInfo&) = default;

  ///
  /// @brief Default move constructor. 
  ///
  CKCInfo(CKCInfo&&) noexcept = default;

  ///
  /// @brief Default move assign operator. 
  ///
  CKCInfo& operator=(CKCInfo&&) noexcept = default;

  ///
  /// @brief Name of the first contact frame.
  ///
  std::string frame_0;

  ///
  /// @brief Name of the contact frame.
  /// 
  std::string frame_1;



  ///
  /// @brief The position gain of the Baumgarte's stabilization method. 
  /// Default is 0.0.
  ///
  double baumgarte_position_gain = 0.0;

  ///
  /// @brief The velocity gain of the Baumgarte's stabilization method. 
  /// Default is 0.0.
  ///
  double baumgarte_velocity_gain = 0.0;

  
};

} // namespace robotoc

#endif // ROBOTOC_CONTACT_MODEL_INFO_HPP_