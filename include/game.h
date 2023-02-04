#ifndef GAME_H
#define GAME_H

#include "map.h"
#include "list.h"

typedef struct  s_game_infos
{
	t_list		clients;
	t_map		rowid_to_user;
	//More infos will be added
}				t_game_infos;

typedef struct  s_user
{

}				t_user;

typedef struct  s_character
{

}				t_character;

//client->row_id_user = -1 => On recupere la donnée en cas de connexion, on ajoute un element dans le t_map characters (key = row_id_user, value = NULL) et on renvoie la list des personnages du user
//Si un autre client tente une connexion => on peut le savoir en regardant dans le t_map




void* search_and_compute_tasks(void* datas);

#endif
