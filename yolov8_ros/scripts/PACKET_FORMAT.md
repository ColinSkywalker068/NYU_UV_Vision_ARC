# STM32 Serial Packet Format

## Overview
The STM32 serial communication uses a binary packet format to send auto-aim data (pitch, yaw, and target status) from the CV system to the STM32 microcontroller.

## Packet Structure

The packet consists of three parts: **Header**, **Body**, and **Footer**.

```
┌─────────────────────────────────────────────────────────────┐
│                    COMPLETE PACKET (15 bytes)                 │
├─────────────────────────────────────────────────────────────┤
│ HEADER (7 bytes)                                             │
│ ├─ Magic Byte (1 byte)                                       │
│ ├─ Length (2 bytes)                                          │
│ ├─ Reserved (2 bytes)                                        │
│ └─ Message Type (2 bytes)                                    │
├─────────────────────────────────────────────────────────────┤
│ BODY (9 bytes)                                               │
│ ├─ Pitch (4 bytes, signed int)                                │
│ ├─ Yaw (4 bytes, signed int)                                 │
│ └─ Has Target (1 byte, unsigned)                             │
├─────────────────────────────────────────────────────────────┤
│ FOOTER (2 bytes)                                              │
│ └─ Reserved (2 bytes)                                        │
└─────────────────────────────────────────────────────────────┘
```

## Detailed Breakdown

### Header (7 bytes)
```python
struct.pack("BHBBH", header, length, 0x00, 0x00, msg_type)
```

| Field | Type | Size | Value | Description |
|-------|------|------|-------|-------------|
| **Magic Byte** | `B` (unsigned char) | 1 byte | `0xE7` | Packet start marker/sync byte |
| **Length** | `H` (unsigned short) | 2 bytes | `11` | Length of body section (in bytes) |
| **Reserved 1** | `B` (unsigned char) | 1 byte | `0x00` | Reserved for future use |
| **Reserved 2** | `B` (unsigned char) | 1 byte | `0x00` | Reserved for future use |
| **Message Type** | `H` (unsigned short) | 2 bytes | `1` | Message type identifier (1 = auto-aim) |

**Python struct format:** `"BHBBH"` means:
- `B` = unsigned char (1 byte)
- `H` = unsigned short (2 bytes, little-endian)
- `B` = unsigned char (1 byte)
- `B` = unsigned char (1 byte)
- `H` = unsigned short (2 bytes, little-endian)

### Body (9 bytes)
```python
struct.pack("iiB", pitch_int, yaw_int, has_target)
```

| Field | Type | Size | Value | Description |
|-------|------|------|-------|-------------|
| **Pitch** | `i` (signed int) | 4 bytes | `int(pitch * 1000000)` | Pitch angle in micro-radians (rad × 10⁶) |
| **Yaw** | `i` (signed int) | 4 bytes | `int(yaw * 1000000)` | Yaw angle in micro-radians (rad × 10⁶) |
| **Has Target** | `B` (unsigned char) | 1 byte | `0` or `1` | Target detection flag (1 = target detected, 0 = no target) |

**Python struct format:** `"iiB"` means:
- `i` = signed int (4 bytes, little-endian)
- `i` = signed int (4 bytes, little-endian)
- `B` = unsigned char (1 byte)

**Note:** Angles are multiplied by 1,000,000 to preserve precision when converting from float to integer. This allows sub-micro-radian precision.

### Footer (2 bytes)
```python
struct.pack("H", 0x0)
```

| Field | Type | Size | Value | Description |
|-------|------|------|-------|-------------|
| **Reserved** | `H` (unsigned short) | 2 bytes | `0x0000` | Reserved for future use (checksum, etc.) |

**Python struct format:** `"H"` means:
- `H` = unsigned short (2 bytes, little-endian)

## Example Packet

### Example 1: Zero angles, no target
```
Input:
  pitch = 0.0 rad
  yaw = 0.0 rad
  has_target = 0

Packet (hex):
  E7 0B 00 00 00 01 00   00 00 00 00 00 00 00 00 00   00 00
  └─ Header ─┘            └──── Body ────┘            └Footer┘
  
Byte breakdown:
  Header:  E7 0B 00 00 00 01 00
  Body:    00 00 00 00 00 00 00 00 00
  Footer:  00 00
```

### Example 2: Small angles, target present
```
Input:
  pitch = 0.123456 rad
  yaw = -0.234567 rad
  has_target = 1

Calculations:
  pitch_int = int(0.123456 * 1000000) = 123456
  yaw_int = int(-0.234567 * 1000000) = -234567

Packet (hex):
  E7 0B 00 00 00 01 00   40 E2 01 00 F9 7A FC FF 01   00 00
  └─ Header ─┘            └──── Body ────┘            └Footer┘

Byte breakdown:
  Header:  E7 0B 00 00 00 01 00
  Body:    40 E2 01 00 (pitch=123456)  F9 7A FC FF (yaw=-234567)  01 (has_target=1)
  Footer:  00 00
```

## Total Packet Size
- **Header:** 7 bytes
- **Body:** 9 bytes
- **Footer:** 2 bytes
- **Total:** 15 bytes

## Data Encoding Details

### Angle Encoding
Angles are sent as **signed 32-bit integers** representing **micro-radians**:
- **Range:** -2,147,483,648 to 2,147,483,647 micro-radians
- **Equivalent to:** -2,147.483648 to 2,147.483647 radians
- **Precision:** 1 micro-radian = 0.000001 rad ≈ 0.000057 degrees

**Conversion:**
```python
# Float to integer (for transmission)
pitch_int = int(pitch_radians * 1000000)

# Integer to float (on STM32 side)
pitch_radians = pitch_int / 1000000.0
```

### Endianness
All multi-byte values use **little-endian** byte order (standard on x86/x64 and ARM architectures).

## Protocol Notes

1. **Magic Byte (0xE7):** Used for packet synchronization. The STM32 can use this to detect the start of a new packet.

2. **Length Field:** Indicates the body length (11 bytes). This allows the STM32 to verify packet integrity.

3. **Message Type:** Currently only type `1` (auto-aim) is used. This field allows for future expansion to support different message types.

4. **No Checksum:** The current implementation doesn't include a checksum. The STM32 should validate:
   - Magic byte matches
   - Length field matches expected body size
   - Message type is valid

5. **Synchronization:** If bytes are lost or corrupted, the STM32 should:
   - Wait for the magic byte (0xE7)
   - Read the length field
   - Read the expected number of bytes
   - Validate the structure

## STM32 Side Implementation (Pseudo-code)

```c
// Expected packet structure
typedef struct {
    uint8_t  magic;        // 0xE7
    uint16_t length;        // 11
    uint8_t  reserved1;    // 0x00
    uint8_t  reserved2;    // 0x00
    uint16_t msg_type;     // 1
    int32_t  pitch;         // micro-radians
    int32_t  yaw;           // micro-radians
    uint8_t  has_target;    // 0 or 1
    uint16_t footer;        // 0x0000
} __attribute__((packed)) AutoAimPacket;

// Receive and parse
void parse_auto_aim_packet(uint8_t* buffer) {
    AutoAimPacket* packet = (AutoAimPacket*)buffer;
    
    // Validate magic byte
    if (packet->magic != 0xE7) return;
    
    // Validate length
    if (packet->length != 11) return;
    
    // Validate message type
    if (packet->msg_type != 1) return;
    
    // Convert to radians
    float pitch_rad = packet->pitch / 1000000.0f;
    float yaw_rad = packet->yaw / 1000000.0f;
    
    // Use the data
    if (packet->has_target) {
        // Apply pitch and yaw angles
        apply_angles(pitch_rad, yaw_rad);
    }
}
```

