#ifndef GAME_H
#define GAME_H

#include "map.h"
#include "list.h"

typedef struct
{
	t_list	messages_received;
	t_list	messages_to_send;
}			t_game;

//typedef struct	s_game_infos
//{
//	t_list		messages;
//	//t_map		user_id_to_authentified_client;
//}				t_game_infos;
//
///*
// * /brief function called in multiple threads to consume messages
// *
// * /param[in] datas can be anything, a cast is done ine the function
// *
// * /retval NULL
// */
//void* search_and_compute_tasks(void* datas);

#endif /* GAME_H */
