#ifndef MESSAGE_H
#define MESSAGE_H

#include <stdint.h>

#ifdef linux
#define PACK( __Declaration__ ) __Declaration__ __attribute__((__packed__))
#else
#define PACK( __Declaration__ ) __pragma( pack(push, 1) ) __Declaration__ __pragma( pack(pop))
#endif

/*!
 * \brief Basic message that we send through the network
 */
PACK(struct s_message
{
	uint32_t	size;     /*!< total size of the message */
	uint32_t	type;     /*!< type of the message (used to store an enum value) */
	char		buffer[]; /*!< contains message data */
});

typedef struct s_message t_message;

#endif