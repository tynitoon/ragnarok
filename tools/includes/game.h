#ifndef GAME_H
#define GAME_H

#include "map.h"
#include "list.h"

typedef struct s_game_infos
{
	t_list     clients;
	t_map      user_id_to_authentified_client;
}              t_game_infos;

/*
 * /brief function called in multiple threads to consume messages
 *
 * /param[in] datas can be anything, a cast is done ine the function
 *
 * /retval NULL
 */
void* search_and_compute_tasks(void* datas);

#endif /* GAME_H */