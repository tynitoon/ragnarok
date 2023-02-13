#define WIN32_LEAN_AND_MEAN

#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <stdlib.h>
#include <stdio.h>
#include <io.h>

#include "client.h"
#include "single_memory.h"

// Need to link with Ws2_32.lib, Mswsock.lib, and Advapi32.lib
#pragma comment (lib, "Ws2_32.lib")
#pragma comment (lib, "Mswsock.lib")
#pragma comment (lib, "AdvApi32.lib")


int start_client(char* address, char* port, t_server* server)
{
	WSADATA             wsaData;
	struct addrinfo* possible_address = NULL;
	struct addrinfo     server_address;
	int                 ret_value;
	t_list_element*		save;
	t_message*          message;
	uint64_t			message_size;

	//Init WinSock
	if ((ret_value = WSAStartup(MAKEWORD(2, 2), &wsaData)) != 0)
	{
		printf("WSAStartup failed with error: %d\n", ret_value);
		return -1;
	}

	memset(&server_address, 0, sizeof(server_address));
	server_address.ai_family = AF_INET;
	server_address.ai_socktype = SOCK_STREAM;
	server_address.ai_protocol = IPPROTO_TCP;

	//Resolve the server address and port
	if ((ret_value = getaddrinfo(address, port, &server_address, &possible_address)) != 0)
	{
		printf("getaddrinfo failed with error: %d\n", ret_value);
		WSACleanup();
		return -1;
	}

	//Create socket
	if ((server->fd = socket(possible_address->ai_family, possible_address->ai_socktype, possible_address->ai_protocol)) == INVALID_SOCKET)
	{
		printf("socket failed with error: %ld\n", WSAGetLastError());
		freeaddrinfo(possible_address);
		WSACleanup();
		return -1;
	}

	//Connect to server
	if ((ret_value = connect(server->fd, possible_address->ai_addr, (int)possible_address->ai_addrlen)) == SOCKET_ERROR)
	{
		closesocket(server->fd);
		freeaddrinfo(possible_address);
		WSACleanup();
		return -1;
	}

	while (1)
	{
		//Check if it was for closing and read the incoming message
		if ((ret_value = _read(server->fd, &server->buffer[server->buffer_index], BUFFER_SIZE - server->buffer_index)) == 0)
		{
			//Close the socket
			closesocket(server->fd);
			server->fd = -1;
			WSACleanup();

			return 0;
		}
		else
		{
			server->buffer_index += ret_value;

			//If there is enough data to get the message's size
			while (server->buffer_index >= (int)sizeof(int))
			{
				message = (t_message*)server->buffer;
				message_size = message->size;

				//if the message_size is impossible (cheater or network error)
				if (message_size < sizeof(t_message) || message_size > BUFFER_SIZE)
					server->buffer_index = 0;
				else if (server->buffer_index >= message_size)
				{
					save = (t_list_element*)get_memory(sizeof(t_list_element) + message_size);
					message = (t_message*)save->data;

					//Copy datas
					memcpy(message, server->buffer, message_size);

					//Change the size to have only the data size
					message->size = message_size - sizeof(t_message);
					add_list_element(&server->messages, save);

					//Shift datas in the buffer (circular buffer)
					server->buffer_index -= message_size;
					memmove(server->buffer, &server->buffer[message_size], server->buffer_index);
				}
				else
					break;
			}
		}
	}

	return 0;
}