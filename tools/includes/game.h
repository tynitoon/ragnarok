#ifndef GAME_H
#define GAME_H

#include "map.h"
#include "list.h"

typedef struct  s_game_infos
{
	t_list		clients;
	t_map		user_id_to_authentified_client;
}				t_game_infos;

typedef struct  s_position
{
	int			map;
	float		x;
	float		y;
}				t_position;

typedef struct  s_character
{
	int			user_id;
	char		name[32];
	t_position	position;
}				t_character;

void* search_and_compute_tasks(void* datas);

#endif
