#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "sqlite.h"
#include "game.h"
#include "single_memory.h"
#include "server.h"

#include "map.h"

int main()
{
	t_game_infos	game_infos;
	int				count_threads;

	//Init values
	if (init_database() != 0)
		return 1;

	memset(&game_infos, 0, sizeof(t_game_infos));

	int walah[2];
	walah[0] = 1;
	walah[1] = 2;

	//t_list_element2 tmp;
	//tmp.prev = NULL;
	//tmp.next = NULL;
	//printf("%ld %ld %ld %ld\n", (unsigned long)&tmp.prev, (unsigned long)&tmp.next, (unsigned long)&tmp.buffer, (unsigned long)tmp.buffer);

	sqlite_set_array("UPDATE user SET character_row_ids = ? WHERE username = 'default'", 2, sizeof(int), walah);

	t_sqlite_array* data = sqlite_get_array("SELECT character_row_ids FROM user WHERE username = 'default'");

	if (data != NULL)
	{
		size_t i = 0;
		for (i = 0; i < data->size; ++i)
		{
			printf("value = %d\n", ((int*)data->buffer)[i]);
		}
		free_memory(data);
	}

	//t_map map;
	//
	//init_map(&map, sizeof(int));

	//int j;
	//int arr[100000];

	//for (j = 0; j < 100000; ++j)
	//{
	//	arr[j] = j;
	//	add_map_element(&map, &arr[j], &arr[j]);
	//	//printf("j = %d\n", j);
	//}

	////display_map(&map);
	//display_memory();

	//display_map(&map);


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