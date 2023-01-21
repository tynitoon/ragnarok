#include <stddef.h>
#include <signal.h>
#include <unistd.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>

#include <stdio.h>

#include "game.h"
#include "single_memory.h"
#include "server.h"

static void sig_handler()
{
	release_memory();
	exit(0);
}

void* myThreadFun(void* datas)
{
	t_game_infos* game_infos = (t_game_infos*)datas;

	game_infos = game_infos;
	while (true)
	{
		sleep(1);
	}
	return NULL;
}

int main()
{
	t_game_infos	game_infos;
	int				count_threads;

	//Init values
	memset(&game_infos, 0, sizeof(t_game_infos));
	count_threads = sysconf(_SC_NPROCESSORS_ONLN) - 1; //We remove one for the main thread

	//Handle CTRL+C
	signal(SIGINT, sig_handler);

	//Start thread consummers
	for (int i = 0; i < count_threads; ++i)
	{
		pthread_t	thread_id;

		pthread_create(&thread_id, NULL, myThreadFun, &game_infos);
		pthread_detach(thread_id);
	}

	start_server(4242, &game_infos.clients);
	//send(fd, buffer , strlen(buffer) , 0 );

	return 0;
}