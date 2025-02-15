#include "server.h"

#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <sys/time.h>

#include "message.h"
#include "single_memory.h"

#define BUFFER_SIZE 4096

/*!
 * \brief Contains client info
 */
typedef struct	s_client
{
	int			sockfd;					/*!< file description of the client */
	size_t		nb_bytes;				/*!< number of bytes filled into the buffer */
	char		buffer[BUFFER_SIZE];	/*!< buffer that will be filled when a message is received */
}				t_client;

void server_start(int port, t_list* messages_received, t_list* messages_to_send)
{
	/* Create a TCP master socket */
	int master_socket;
	if ((master_socket = socket(AF_INET, SOCK_STREAM, 0)) == 0)
	{
		perror("socket failed");
		return;
	}

	/* Set socket option so the address can be reused when relaunching the server */
	int opt = 1;
	if (setsockopt(master_socket, SOL_SOCKET, SO_REUSEADDR, (char*)&opt, sizeof(opt)) < 0)
	{
		perror("setsockopt failed");
		return;
	}

	/* Bind the socket to localhost port */
	struct sockaddr_in address;
	bzero(&address, sizeof(address));
	address.sin_family = AF_INET;
	address.sin_addr.s_addr = INADDR_ANY;
	address.sin_port = htons(port);
	if (bind(master_socket, (struct sockaddr*)&address, sizeof(address)) < 0)
	{
		perror("bind failed");
		return;
	}

	/* Specify maximum of 10 pending connections for the master socket */
	if (listen(master_socket, 10) < 0)
	{
		perror("listen");
		return;
	}

	/* Loop for connection and messages */
	fd_set readfds;
	fd_set writefds;
	t_list clients;
	list_init(&clients);

	/* Prepare a timeout for select in order to send message without waiting to receive one */
	struct timeval timeout;
	timeout.tv_sec = 0;
	timeout.tv_usec = 5000;
	while (1)
	{
		/* Clear the socket set */
		FD_ZERO(&readfds);
		FD_ZERO(&writefds);

		/* Add master socket to set */
		FD_SET(master_socket, &readfds);

		/* Add read sockets and find max_fd */
		int max_sockfd = master_socket;
		t_list_element* tmp = clients.head;
		int addrlen = sizeof(struct sockaddr_in);
		while (tmp != NULL)
		{
			t_list_element *save_next = tmp->next;
			t_client* client = (t_client*)tmp->data;
			int sockfd = client->sockfd;

			if (sockfd > 0)
			{
				FD_SET(sockfd, &readfds);
				if (sockfd > max_sockfd)
					max_sockfd = sockfd;
			}
			else /* Client is not connected anymore */
				FREE(list_remove(&clients, tmp->data));

			tmp = save_next;
		}

		/* Add write sockets */
		mutex_lock(&messages_to_send->mutex);
		tmp = messages_to_send->head;
		while (tmp != NULL)
		{
			t_dest_message* msg = ((t_list*)tmp->data)->head->data;
			FD_SET(msg->sockfd, &writefds);
			tmp = tmp->next;
		}
		mutex_unlock(&messages_to_send->mutex);

		/* Wait for an activity on one of the sockets */
		if ((select(max_sockfd + 1, &readfds, &writefds, NULL, &timeout) < 0) && (errno != EINTR))
			fprintf(stderr, "Error in start_server: select failed\n");

		/* If something happened on the master socket then its an incoming connection */
		if (FD_ISSET(master_socket, &readfds))
		{
			int sockfd = accept(master_socket, (struct sockaddr*)&address, (socklen_t*)&addrlen);
			if (sockfd < 0)
			{
				perror("accept");
				return;
			}

			printf("New connection , socket fd is %d , ip is : %s , port : %d\n", sockfd, inet_ntoa(address.sin_addr), ntohs(address.sin_port));

			/* Create new client */
			t_client* client = MALLOC(sizeof(t_client));
			memset(client, 0, sizeof(t_client));
			client->sockfd = sockfd;
			list_add_back(&clients, client);
		}

		/* Check if there are incoming messages */
		tmp = clients.head;
		while (tmp != NULL)
		{
			t_client* client = (t_client*)tmp->data;
			int sockfd = client->sockfd;
			if (FD_ISSET(sockfd, &readfds))
			{
				int ret_value = read(sockfd, &client->buffer[client->nb_bytes], BUFFER_SIZE - client->nb_bytes);
				/* Check if it was for close and read the incoming message */
				if (ret_value == 0)
				{
					/* A client is disconnected, get his details and print */
					getpeername(sockfd, (struct sockaddr*)&address, (socklen_t*)&addrlen);
					printf("client disconnected , ip %s , port %d \n", inet_ntoa(address.sin_addr), ntohs(address.sin_port));

					/* Close the socket */
					close(sockfd);

					client->sockfd = -1;
				}
				else
				{
					size_t offset = 0;
					client->nb_bytes += ret_value;
					printf("ret_value = %d\n", ret_value);

					/* If there is enough data to get the message's size */
					while (client->nb_bytes - offset >= sizeof(t_message))
					{
						t_message *message = (t_message*)&client->buffer[offset];
						printf("message size = %d\n", message->size);
						/* Flush data if the message_size is impossible (cheater or network error) */
						if (message->size > BUFFER_SIZE)
							client->nb_bytes = 0;
						else if (client->nb_bytes - offset >= message->size)
						{
							/* Duplicate data and add message */
							t_dest_message *new_message = MALLOC(sizeof(int) + message->size);
							new_message->sockfd = sockfd;
							memcpy(&new_message->message, &client->buffer[offset], message->size);

							list_add_back(messages_received, new_message);
							printf("message received : %s\n", new_message->message.buffer);

							/* Shift datas in the buffer(circular buffer) */
							offset += message->size;
						}
						else
							break;
					}
					client->nb_bytes -= offset;
					memmove(client->buffer, &client->buffer[offset], client->nb_bytes);
				}
			}
			tmp = tmp->next;
		}

		/* Check if we can write our messages */
		mutex_lock(&messages_to_send->mutex);
		tmp = messages_to_send->head;
		while (tmp != NULL)
		{
			t_list_element *save_next = tmp->next;
			t_dest_message *msg = ((t_list*)tmp->data)->head->data;
			if (FD_ISSET(msg->sockfd, &writefds))
			{
				while ((msg = list_remove_front(tmp->data)) != NULL)
				{
					printf("Send message : %s to %d size = %d\n", msg->message.buffer, msg->sockfd, msg->message.size);
					if (send(msg->sockfd, &msg->message, msg->message.size, 0) < 0)
						perror("send");
					FREE(msg);
				}
				FREE(list_remove(messages_to_send, tmp->data));
			}
			tmp = save_next;
		}
		mutex_unlock(&messages_to_send->mutex);
	}
}
