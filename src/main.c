#include <unistd.h>
#include <string.h>

#include "game.h"
#include "single_memory.h"
#include "server.h"

#include "sqlite/sqlite3.h"

int main()
{
	t_game_infos	game_infos;
	int				count_threads;

	sqlite3* db;

	sqlite3_open("database.db", &db);

	//Init values
	if (init_list(&game_infos.clients) != 0)
		return 1;

	count_threads = sysconf(_SC_NPROCESSORS_ONLN) - 1; //We remove one for the main thread

	//Start thread consummers
	for (int i = 0; i < count_threads; ++i)
	{
		pthread_t	thread_id;

		pthread_create(&thread_id, NULL, search_and_compute_tasks, &game_infos);
		pthread_detach(thread_id);
	}

	start_server(4242, &game_infos.clients);

	return 0;
}