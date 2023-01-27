#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "sqlite.h"
#include "game.h"
#include "single_memory.h"
#include "server.h"

#include "hash.h"
#include "map.h"

int main()
{
	t_game_infos	game_infos;
	int				count_threads;

	//Init values
	if (init_database() != 0 ||
		init_list(&game_infos.clients) != 0)
		return 1;

	//int walah[2];
	//walah[0] = 1;
	//walah[1] = 2;

	//t_list_element2 tmp;
	//tmp.prev = NULL;
	//tmp.next = NULL;
	//printf("%ld %ld %ld %ld\n", (unsigned long)&tmp.prev, (unsigned long)&tmp.next, (unsigned long)&tmp.buffer, (unsigned long)tmp.buffer);

	//sqlite_set_array("UPDATE user SET character_row_ids = ? WHERE username = 'default'", 2, sizeof(int), walah);

	//t_sqlite_array* data = sqlite_get_array("SELECT character_row_ids FROM user WHERE username = 'default'");

	//if (data != NULL)
	//{
	//	size_t i = 0;
	//	for (i = 0; i < data->size; ++i)
	//	{
	//		printf("value = %d\n", ((int*)data->buffer)[i]);
	//	}
	//	free_memory(data);
	//}

	//t_map map;
	//init_map(&map);

	//int i;

	//for (i = 0; i < 6; ++i)
	//{
	//	add_map_element(&map, i, &arr[i]);
	//	display_map(&map);
	//	printf("_____________\n");
	//}


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