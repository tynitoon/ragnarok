#pragma comment (lib, "Ws2_32.lib")
#pragma comment (lib, "Mswsock.lib")
#pragma comment (lib, "AdvApi32.lib")

#include "client.h"

#include "list.h"
#include <winsock2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "single_memory.h"
#include "message.h"

#define BUFFER_SIZE 4096

void client_start(char* ip, int port, t_list* messages_received, t_list* messages_to_send)
{
	/* Initialisation de Winsock */
	WSADATA wsaData;
	if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0)
	{
		perror("WSAStartup failed");
		return;
	}

	/* Création du socket */
	SOCKET sockfd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	if (sockfd == INVALID_SOCKET)
	{
		perror("Error creating socket");
		WSACleanup();
		return;
	}

	/* Configuration de l'adresse du serveur */
	struct sockaddr_in server_addr;
	memset(&server_addr, 0, sizeof(server_addr));
	server_addr.sin_family = AF_INET;
	server_addr.sin_port = htons(port);
	server_addr.sin_addr.s_addr = inet_addr(ip);

	/* Connexion au serveur */
	if (connect(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr)) == SOCKET_ERROR)
	{
		perror("Error connecting");
		closesocket(sockfd);
		WSACleanup();
		return;
	}

	fd_set read_fds;
	fd_set write_fds;
	char buffer[BUFFER_SIZE];
	size_t nb_bytes = 0;
	while (1)
	{
		FD_ZERO(&read_fds);
		FD_ZERO(&write_fds);
		FD_SET(sockfd, &read_fds);
		if (!list_is_empty(messages_to_send))
			FD_SET(sockfd, &write_fds);

		int ready = select(0, &read_fds, &write_fds, NULL, NULL);

		if (ready == SOCKET_ERROR)
		{
			perror("Error in select");
			break;
		}

		if (ready > 0)
		{
			if (FD_ISSET(sockfd, &write_fds))
			{
				// Envoi de messages
				t_message *message = list_remove_front(messages_to_send);
				if (message != NULL)
				{
					if (send(sockfd, message, message->size, 0) == SOCKET_ERROR)
						perror("Error sending message content");
					printf("message is sent : %s size = %d calculated size = %d\n", message->buffer, message->size);
					FREE(message);
				}
			}

			if (FD_ISSET(sockfd, &read_fds))
			{
				// Réception de données
				int bytes_received = recv(sockfd, &buffer[nb_bytes], BUFFER_SIZE - nb_bytes, 0);
				if (bytes_received == SOCKET_ERROR)
					perror("Error receiving");
				else if (bytes_received == 0)
				{
					printf("Connexion fermée par le serveur.\n");
					break;
				}
				else
				{
					size_t offset = 0;
					nb_bytes += bytes_received;

					printf("bytes received = %d\n", bytes_received);
					/* If there is enough data to get the message's size */
					while (nb_bytes - offset >= sizeof(t_message))
					{
						t_message* message = (t_message*)&buffer[offset];
						printf("size message = %d\n", message->size);
						/* Flush data if the message_size is impossible (cheater or network error) */
						if (message->size > BUFFER_SIZE)
							nb_bytes = 0;
						else if (nb_bytes - offset >= message->size)
						{
							/* Duplicate data and add message */
							t_message *new_message = MALLOC(message->size);
							memcpy(new_message, &buffer[offset], message->size);
							printf("message received = %s\n", new_message->buffer);
							list_add_back(messages_received, new_message);

							/* Shift datas in the buffer(circular buffer) */
							offset += message->size;
						}
						else
							break;
					}
					nb_bytes -= offset;
					memmove(buffer, &buffer[offset], nb_bytes);
				}
			}
		}
	}

	closesocket(sockfd);
	WSACleanup();
}