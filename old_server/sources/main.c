#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "single_memory.h"
#include "game.h"
#include "server.h"

void add_send_message(t_list *list, t_dest_message *message)
{
	mutex_lock(&list->mutex);

	/* Search if we already have a list with this sockfd */
	t_list_element *tmp = list->head;
	while (tmp != NULL)
	{
		t_dest_message *msg = ((t_list*)tmp->data)->head->data;
		if (msg->sockfd == message->sockfd)
		{
			list_add_back(tmp->data, message);
			mutex_unlock(&list->mutex);
			return;
		}
		tmp = tmp->next;
	}

	/* Create new list corresponding to the new sockfd */
	t_list *new_list = MALLOC(sizeof(t_list));
	list_init(new_list);
	list_add_back(new_list, message);
	list_add_back(list, new_list);

	mutex_unlock(&list->mutex);
}

void* test_read_write(void* data)
{
	t_game* game = data;
	while (1)
	{
		t_dest_message *msg;
		while ((msg = list_remove_front(&game->messages_received)) != NULL)
		{
			printf("sockfd = %d, message = %s\n", msg->sockfd, msg->message.buffer);
			int size_word = strlen("hello");
			t_dest_message *new_msg = MALLOC(sizeof(t_dest_message) + size_word + 1);
			new_msg->sockfd = msg->sockfd;
			new_msg->message.size = sizeof(t_message) + size_word + 1;
			strcpy(new_msg->message.buffer, "hello");
			add_send_message(&game->messages_to_send, new_msg);
		}
	}
	return NULL;
}

int main()
{
	//Init values
	//t_game_infos game_infos;
	//memset(&game_infos, 0, sizeof(t_game_infos));
	//map_init(&game_infos.user_id_to_authentified_client);
	//if (init_database() != 0)
	//	return 1;

	////Start thread consummers
	//int count_threads = sysconf(_SC_NPROCESSORS_ONLN) - 1; //We remove one for the main thread
	//for (int i = 0; i < count_threads; ++i)
	//{
	//	pthread_t	thread_id;

	//	pthread_create(&thread_id, NULL, search_and_compute_tasks, &game_infos);
	//	pthread_detach(thread_id);
	//}

	t_game game;
	list_init(&game.messages_received);
	list_init(&game.messages_to_send);

	pthread_t	thread_id;

	pthread_create(&thread_id, NULL, test_read_write, &game);
	pthread_detach(thread_id);

	server_start(4242, &game.messages_received, &game.messages_to_send);

	return 0;
}