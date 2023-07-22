#ifndef SQLITE_H
#define SQLITE_H

#include "sqlite3.h"
#include "protocol.h"

typedef struct	s_sqlite_array
{
	size_t		size;
	char		buffer[];
}				t_sqlite_array;

/*
 * /brief Initialize the database by opening or creating a .db file. This function has to be call before any sqlite function
 */
int				init_database();

/*
 * /brief unused function at the moment (used as example)
 */
int				sqlite_set_array(char* sql_command, size_t array_size, size_t element_size, void* array);

/*
 * /brief unused function at the moment (used as example)
 */
t_sqlite_array* sqlite_get_array(char* sql_command);

/*
 * /brief unused function at the moment (used as example)
 */
int				sqlite_get_integer(char* sql_command);

/*
 * /brief Retrieve characters from database
 *
 * /param[in] sql_command is the sql command to select the corresponding characters
 * /param[out] character_counter is an integer that will be fill to give back the amount of characters in the returned array
 * 
 * /return an array of character 
 */
t_character*	sqlite_get_characters(char* sql_command, int* character_counter);

#endif /* SQLITE_H */