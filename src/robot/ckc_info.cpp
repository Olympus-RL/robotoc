#include "robotoc/robot/ckc_info.hpp"

#include <stdexcept>

namespace robotoc {

CKCInfo::CKCInfo(const std::string& _frame_0, 
                const std::string& _frame_1, 
                const double _baumgarte_time_step)
  : frame_0(_frame_0),
    frame_1(_frame_1),
    baumgarte_position_gain(1.0/(_baumgarte_time_step*_baumgarte_time_step)),
    baumgarte_velocity_gain(2.0/(_baumgarte_time_step)) {
  if (_baumgarte_time_step < 1e-10) {
    throw std::out_of_range("[CKC_INFO] invalid argument: 'baumgarte_time_step' must be positive!");
  }
}


CKCInfo::CKCInfo(const std::string& _frame_0,
                const std::string& _frame_1,
                const double _baumgarte_position_gain,
                const double _baumgarte_velocity_gain)
  : frame_0(_frame_0),
    frame_1(_frame_1),
    baumgarte_position_gain(_baumgarte_position_gain),
    baumgarte_velocity_gain(_baumgarte_velocity_gain) {
  if (_baumgarte_position_gain < -1e-10) {
    throw std::out_of_range("[CKC_INFO] invalid argument: 'baumgarte_position_gain' must be positive!");
  }
  if (_baumgarte_velocity_gain < -1e-10) {
    throw std::out_of_range("[CKC_INFO] invalid argument: 'baumgarte_velocity_gain' must be positive!");
  }
}

} // namespace robotoc