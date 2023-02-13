#ifndef PROTOCOL_H
#define PROTOCOL_H

#include <stdint.h>

typedef enum				e_data_type
{
	CONNECT					= 0,
	DISCONNECT				= 1,
	MOVE					= 2,
	MESSAGE					= 3,
	POPUP					= 4,
	CHARACTER_LIST_CONNECT	= 5,
	MAX_VALUE				= 0xFFFFFFFFFFFFFFFF
}							t_data_type;

typedef struct  s_message
{
	uint64_t    size;
	t_data_type type;
	char        buffer[];
}               t_message;

typedef struct	s_connect
{
	char		username[32];
	char		password[32];
}				t_connect;

#endif