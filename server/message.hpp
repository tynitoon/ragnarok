#ifndef MESSAGE_HPP
#define MESSAGE_HPP

#include <cstdint>

/**
 * @brief Structure pour stocker un message.
 */
__pragma(pack(push, 1))
struct Message
{
	uint32_t size;
	uint32_t type;
	char data[];
};
__pragma(pack(pop))

#endif