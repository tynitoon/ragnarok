#ifndef SQLITE_H
#define SQLITE_H

#include "sqlite3.h"
#include "protocol.h"

typedef struct	s_sqlite_array
{
	size_t		size;
	char		buffer[];
}				t_sqlite_array;

int				init_database();
int				sqlite_set_array(char* sql_command, size_t array_size, size_t element_size, void* array);
t_sqlite_array* sqlite_get_array(char* sql_command);
int				sqlite_get_integer(char* sql_command);
t_character*	sqlite_get_characters(char* sql_command, int* character_counter);

#endif
