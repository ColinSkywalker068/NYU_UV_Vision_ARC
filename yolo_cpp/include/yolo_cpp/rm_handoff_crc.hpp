#ifndef YOLO_CPP__RM_HANDOFF_CRC_HPP_
#define YOLO_CPP__RM_HANDOFF_CRC_HPP_

#include <cstdint>

namespace yolo_cpp::rm_handoff::crc8 {

uint8_t get_crc8_check_sum(const uint8_t *data, uint32_t len);
void append_crc8_check_sum(uint8_t *data, uint32_t len);

} // namespace yolo_cpp::rm_handoff::crc8

namespace yolo_cpp::rm_handoff::crc16 {

uint16_t get_crc16_check_sum(const uint8_t *data, uint32_t len, uint16_t crc);
bool verify_crc16_check_sum(const uint8_t *data, uint32_t len);
void append_crc16_check_sum(uint8_t *data, uint32_t len);

} // namespace yolo_cpp::rm_handoff::crc16

#endif // YOLO_CPP__RM_HANDOFF_CRC_HPP_
