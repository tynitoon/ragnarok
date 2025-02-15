#ifndef TEST_H
#define TEST_H

#include "list.h"

typedef struct
{
	t_list messages_received;
	t_list messages_to_send;
} t_game;

void start_thread(t_game* game);

#endif