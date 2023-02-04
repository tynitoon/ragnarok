#ifndef PROTOCOL_H
#define PROTOCOL_H

typedef enum    e_data_type
{
	CONNECT		= 0,
	DISCONNECT	= 1,
	MOVE		= 2,
	MESSAGE		= 3,
	POPUP		= 4,
	MAX_VALUE	= 0xFFFFFFFFFFFFFFFF
}               t_data_type;

typedef struct  s_message
{
	size_t      size;
	t_data_type type;
	char        buffer[];
}               t_message;

typedef struct	s_connect
{
	char		username[32];
	char		password[32];
}				t_connect;

#endif