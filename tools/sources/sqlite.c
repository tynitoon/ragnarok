#include <stdio.h>
#include <string.h>

#include "sqlite.h"
#include "single_memory.h"

#define PATH_TO_DATABASE "./datas/database.db"

static sqlite3* database = NULL;

int init_database()
{
	if (sqlite3_open(PATH_TO_DATABASE, &database) != SQLITE_OK)
	{
		fprintf(stderr, "Error in sqlite_init: Cannot open database: %s\n", sqlite3_errmsg(database));
		sqlite3_close(database);
		return -1;
	}

	return 0;
}

int sqlite_set_array(char* sql_command, uint64_t array_size, uint64_t element_size, void* array)
{
	sqlite3_stmt*	sql_statement;
	t_sqlite_array* data;
	uint64_t		full_size;

	full_size = sizeof(t_sqlite_array) + array_size * element_size;

	if (sqlite3_prepare_v2(database, sql_command, -1, &sql_statement, 0) != SQLITE_OK)
	{
		fprintf(stderr, "Error in sqlite_set_array: Cannot prepare statement: %s\n", sqlite3_errmsg(database));
		return -1;
	}

	data = memory_get(full_size);
	data->size = array_size;
	memcpy(data->buffer, array, array_size * element_size);

	sqlite3_bind_blob(sql_statement, 1, data, full_size, SQLITE_STATIC);

	if (sqlite3_step(sql_statement) != SQLITE_DONE) //Entry doesn't exist
	{
		fprintf(stderr, "Error in sqlite_set_array: Execution failed: %s", sqlite3_errmsg(database));
		memory_free(data);
		return -1;
	}

	memory_free(data);
	sqlite3_finalize(sql_statement);

	return 0;
}

t_sqlite_array *sqlite_get_array(char* sql_command)
{
	sqlite3_stmt*	sql_statement;
	t_sqlite_array* data;
	uint64_t		full_size;

	if (sqlite3_prepare_v2(database, sql_command, -1, &sql_statement, 0) != SQLITE_OK)
	{
		fprintf(stderr, "Error in sqlite_get_array: Cannot prepare statement: %s\n", sqlite3_errmsg(database));
		return NULL;
	}

    if (sqlite3_step(sql_statement) != SQLITE_ROW) //Entry doesn't exist
		return NULL;

	full_size = (uint64_t)sqlite3_column_bytes(sql_statement, 0);
	data = memory_get(full_size);
	memcpy(data, sqlite3_column_blob(sql_statement, 0), full_size);

    sqlite3_finalize(sql_statement);

	return data;
}

int	sqlite_get_integer(char* sql_command)
{
	sqlite3_stmt*	sql_statement;
	int				data;

	if (sqlite3_prepare_v2(database, sql_command, -1, &sql_statement, 0) != SQLITE_OK)
	{
		fprintf(stderr, "Error in sqlite_get_integer: Cannot prepare statement: %s\n", sqlite3_errmsg(database));
		return -1;
	}

	if (sqlite3_step(sql_statement) != SQLITE_ROW) //Entry doesn't exist
		return -1;

	data = sqlite3_column_int(sql_statement, 0);

	sqlite3_finalize(sql_statement);

	return data;
}

t_character* sqlite_get_characters(char* sql_command, int* character_counter)
{
	t_character*	characters = NULL;
	sqlite3_stmt*	sql_statement;
	uint64_t		count = 0;
	uint64_t		allocated_size = 0;
	uint64_t		size;

	if (sqlite3_prepare_v2(database, sql_command, -1, &sql_statement, 0) != SQLITE_OK)
	{
		fprintf(stderr, "Error in sqlite_get_characters: Cannot prepare statement: %s\n", sqlite3_errmsg(database));
		return NULL;
	}

	while (sqlite3_step(sql_statement) == SQLITE_ROW)
	{
		if (allocated_size <= count)
		{
			characters = memory_realloc(characters, allocated_size + sizeof(t_character) * 10);
			allocated_size += 10;
		}
		
		//Get user_id
		characters[count].user_id = sqlite3_column_int(sql_statement, 0);

		//Get name
		size = (uint64_t)sqlite3_column_bytes(sql_statement, 1);
		memcpy(characters[count].name, sqlite3_column_text(sql_statement, 1), size);

		//Get position
		size = (uint64_t)sqlite3_column_bytes(sql_statement, 2);
		if (size != 0)
			memcpy(&characters[count].position, sqlite3_column_blob(sql_statement, 2), size);
		else
			memset(&characters[count].position, 0, sizeof(t_position));

		++count;
	}

	*character_counter = count;
	characters = memory_realloc(characters, count * sizeof(t_character));

	sqlite3_finalize(sql_statement);

	return characters;
}

