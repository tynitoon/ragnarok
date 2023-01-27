#include <pthread.h>
#include <unistd.h>

#include "single_memory.h"
#include "protocol.h"
#include "server.h"
#include "game.h"
#include "list.h"

static void connect_command(int fd, t_connect_message* data)
{
	//CREATE MAP COLLECTION
	//CHECK SQLITE PERFORMANCE
	//ADD SQLITE IMPLEMENTATION

	fd = fd;
	data = data;
	//Check that the player is not already connected
	//Get character from sqlite
	//Or create if it doesn't exist
	//Create character structure then add it to map
	//Verify that his position is still possible (move him it's not)
	//Send datas : send(fd, buffer, strlen(buffer), 0 );
}

static void disconnect_command(t_client* client)
{
	//Try to get the character from map
	//Delete character from list
	//Save last infos in sqlite
	client->state = READY_TO_BE_REMOVED;
}

static void handle_message(t_client* client, t_message* message)
{
	switch (message->type)
	{
		case CONNECT:
			connect_command(client->fd, (t_connect_message*)message->buffer);
			break;
		case DISCONNECT:
			disconnect_command(client);
			break;
		case MOVE:
			break;
		case MESSAGE:
			break;
		default:
			break;
	}
}

void* search_and_compute_tasks(void* datas)
{
	t_game_infos*	game_infos = (t_game_infos*)datas;
	t_list_element* client_element;
	t_list_element* message_element;
	t_client*		client;

	while (1)
	{
		client_element = game_infos->clients.head;
		while (client_element != NULL)
		{
			client = (t_client*)client_element->buffer;
			//If there are some messages, lock the client and consume messages
			if (client->state != READY_TO_BE_REMOVED && client->messages.head != NULL && pthread_mutex_trylock(&client->mutex) == 0)
			{
				while (client->messages.head != NULL)
				{
					message_element = remove_list_element(&client->messages, client->messages.head);
					handle_message(client, (t_message*)message_element->buffer);
					free_memory(message_element);
				}
			}
			client_element = client_element->next;
		}
	}
	return NULL;
}