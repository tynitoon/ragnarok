#include <pthread.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>

#include "single_memory.h"
#include "protocol.h"
#include "server.h"
#include "game.h"
#include "list.h"
#include "map.h"
#include "sqlite.h"

static void connect_command(t_game_infos* game_infos, int fd, t_connect* data)
{
	t_user*		user;
	t_message*	message;
	char		buffer[512];
	int			rowid;
	
	//Check username and password
	sprintf(buffer, "SELECT rowid FROM user WHERE username = '%s' AND password = '%s'", data->username, data->password);
	rowid = sqlite_get_integer(buffer);

	//Use buffer to send response
	message = (t_message*)buffer;

	//User doesn't exist or failed his password
	if (rowid == -1)
	{
		message->type = POPUP;
		message->size = sizeof(t_message) + 128 * sizeof(char);
		strcpy(message->buffer, "Error in username or password");
		send(fd, message, message->size, 0);
		return;
	}

	user = get_map_element(game_infos->rowid_to_user, rowid);

	if (user != NULL)
	{
		//User already connected
		//On recupere le client via le user
		//On envoie un message de déco au client
		//On supprime le user.
	}

	//On creer un user
	user = get_memory(sizeof(t_user));
	//On fait les links entre user et client
	//On ajoute le user à la map
	
}

static void disconnect_command(t_client* client)
{
	//Try to get the character from map
	//Delete character from list
	//Save last infos in sqlite
	client = client;
}

static void handle_message(t_game_infos* game_infos, t_client* client, t_message* message)
{
	switch (message->type)
	{
		case CONNECT:
			connect_command(game_infos, client->fd, (t_connect*)message->buffer);
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
	t_client*		client;

	while (1)
	{
		client_element = game_infos->clients.head;
		while (client_element != NULL)
		{
			client = (t_client*)client_element->data;
			//If there are some messages, lock the client and consume messages
			if (client->messages.head != NULL && pthread_mutex_trylock(&client->mutex) == 0)
			{
				while (client->messages.head != NULL)
				{
					handle_message(game_infos, client, (t_message*)client->messages.head->data);
					free_memory(remove_list_element(&client->messages, client->messages.head));
				}

				pthread_mutex_unlock(&client->mutex);
			}
			client_element = client_element->next;
		}
	}
	return NULL;
}