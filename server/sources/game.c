#include <pthread.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>

//#include "single_memory.h"
//#include "protocol.h"
//#include "server.h"
//#include "game.h"
//#include "list.h"
//#include "map.h"
//#include "sqlite.h"
//
//static void connect_command(t_game_infos* game_infos, t_client* client, t_connect* data)
//{
//	(void)game_infos;
//	(void)client;
//	(void)data;
//	////Check username and password
//	//char buffer[512];
//	//sprintf(buffer, "SELECT user_id FROM user WHERE username = '%s' AND password = '%s'", data->username, data->password);
//	//client->user_id = sqlite_get_integer(buffer);
//
//	////Use buffer to send response
//	//t_message* message = (t_message*)buffer;
//
//	////User doesn't exist or failed his password
//	//if (client->user_id == -1)
//	//{
//	//	message->type = POPUP;
//	//	message->size = sizeof(t_message) + 128 * sizeof(char);
//	//	strcpy(message->buffer, "Error in username or password");
//	//	write(client->fd, message, message->size);
//	//	return;
//	//}
//
//	////Check and disconnect if a user is already connected with this username
//	//t_client* already_connected_client = map_get(&game_infos->user_id_to_authentified_client, client->user_id);
//	//if (already_connected_client != NULL)
//	//	close(already_connected_client->fd);
//
//	//map_add(&game_infos->user_id_to_authentified_client, client->user_id, client);
//	//
//	////Get characters then send them to client
//	//int character_counter;
//	//sprintf(buffer, "SELECT user_id, name, position FROM character WHERE user_id = '%d'", client->user_id);
//	//t_character* characters = sqlite_get_characters(buffer, &character_counter);
//	//message->type = CHARACTER_LIST_CONNECT;
//	//message->size = sizeof(t_message) + character_counter * sizeof(t_character);
//	//memcpy(message->buffer, &characters, character_counter * sizeof(t_character));
//	//write(client->fd, message, message->size);
//}
//
//static void disconnect_command(t_client* client)
//{
//	//Try to get the character from map
//	//Delete character from list
//	//Save last infos in sqlite
//	client = client;
//}
//
//static void handle_message(t_game_infos* game_infos, t_client* client, t_message* message)
//{
//	switch (message->type)
//	{
//		case CONNECT:
//			connect_command(game_infos, client, (t_connect*)message->buffer);
//			break;
//		case DISCONNECT:
//			disconnect_command(client);
//			break;
//		case MOVE:
//			break;
//		case MESSAGE:
//			break;
//		default:
//			break;
//	}
//}
//
//void* search_and_compute_tasks(void* datas)
//{
//	//t_game_infos*	game_infos = (t_game_infos*)datas;
//	//t_list_element* client_element;
//	//t_client*		client;
//
//	//while (1)
//	//{
//	//	client_element = game_infos->clients.head;
//	//	while (client_element != NULL)
//	//	{
//	//		client = (t_client*)client_element->data;
//	//		//If there are some messages, lock the client and consume messages
//	//		if (pthread_mutex_trylock(&client->mutex) == 0)
//	//		{
//	//			while ( (t_message *message = list_remove_front(&client->messages)) != NULL)
//	//				handle_message(game_infos, client, message);
//
//	//			pthread_mutex_unlock(&client->mutex);
//	//		}
//	//		client_element = client_element->next;
//	//	}
//	//}
//	return NULL;
//}