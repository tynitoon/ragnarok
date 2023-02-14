#ifndef GAME_H
#define GAME_H

#include "map.h"
#include "list.h"

typedef struct  s_game_infos
{
	t_list		clients;
	t_map		user_id_to_authentified_client;
}				t_game_infos;

void* search_and_compute_tasks(void* datas);

#endif
