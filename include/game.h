#ifndef GAME_H
#define GAME_H

#include "list.h"

typedef struct  s_game_infos
{
	t_list		clients;
	//More infos will be added
}				t_game_infos;

void* search_and_compute_tasks(void* datas);

#endif
