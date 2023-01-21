#ifndef PROTOCOL_H
#define PROTOCOL_H

typedef enum    e_data_type
{
	MESSAGE = 0,
}               t_data_type;

typedef struct  s_message
{
	size_t      size;
	t_data_type type;
	char        buffer[];
}               t_message;

#endif