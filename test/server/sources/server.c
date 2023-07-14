/* //////////////////////////////////////////////////////////////////////// */
/* //////////////////////////////////////////////////////////////////////// */
/* //////////////////////////////////////////////////////////////////////// */
/* THIS FILE IS A COPY, BE CAREFULL TO UPDATE IT FROM THE ORIGINAL SERVER.H */
/* //////////////////////////////////////////////////////////////////////// */
/* //////////////////////////////////////////////////////////////////////// */
/* //////////////////////////////////////////////////////////////////////// */

#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <sys/time.h>
#include <stdlib.h>

#include "list.h"
#include "protocol.h"
#include "server.h"
#include "single_memory.h"

static long int get_timestamp_microsecond()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000000 + tv.tv_usec;
}

int start_server(int port, t_list* clients)
{
	//Create a master socket
	int master_socket;
	if ((master_socket = socket(AF_INET, SOCK_STREAM, 0)) == 0)
	{
		perror("socket failed");
		return -1;
	}

	//Set socket option so the address can be reused when relaunching the server
	int opt = 1;
	if (setsockopt(master_socket, SOL_SOCKET, SO_REUSEADDR, (char*)&opt, sizeof(opt)) < 0)
	{
		perror("setsockopt failed");
		return -1;
	}

	//Bind the socket to localhost port
	struct sockaddr_in address;
	address.sin_family = AF_INET;
	address.sin_addr.s_addr = INADDR_ANY;
	address.sin_port = htons(port);
	if (bind(master_socket, (struct sockaddr*)&address, sizeof(address)) < 0)
	{
		perror("bind failed");
		return -1;
	}

	//Try to specify maximum of 10 pending connections for the master socket
	if (listen(master_socket, 10) < 0)
	{
		perror("listen");
		return -1;
	}

	//Loop for connection and messages
	int             addrlen = sizeof(struct sockaddr_in);
	int             ret_value;
	int             max_fd;
	int             fd;
	t_list_element* tmp;
	t_list_element* save;
	t_client*       client;
	t_message*      message;
	uint64_t        message_size;
	fd_set          readfds;
	int             count_message = -1;
	long int        before = 0;
	long int        duration;
	while (1)
	{
		//Clear the socket set
		FD_ZERO(&readfds);

		//Add master socket to set
		FD_SET(master_socket, &readfds);

		//Add client socket and find max_fd
		max_fd = master_socket;
		tmp = clients->head;
		while (tmp != NULL)
		{
			client = (t_client*)tmp->data;
			fd = client->fd;

			if (fd > 0)
				FD_SET(fd, &readfds);
			else if (client->messages.head == NULL) //Client is not connected anymore and threads has consummed all messages
			{
				save = remove_list_element(clients, tmp); //No problem on thread safe, because the ->next is pointing on the right target
				tmp = save->next;
				free_memory(save); //I don't think it's possible to have a free, then a malloc, then a write memory while a thread is just doing 'if (data == NULL) tmp = tmp->next'

				continue;
			}

			if (fd > max_fd)
				max_fd = fd;

			tmp = tmp->next;
		}

		//Wait for an activity on one of the sockets (timeout is NULL)
		if ((select(max_fd + 1, &readfds, NULL, NULL, NULL) < 0) && (errno != EINTR))
			fprintf(stderr, "Error in start_server: select failed\n");

		//If something happened on the master socket then its an incoming connection
		if (FD_ISSET(master_socket, &readfds))
		{
			if ((fd = accept(master_socket, (struct sockaddr*)&address, (socklen_t*)&addrlen)) < 0)
			{
				perror("accept");
				return -1;
			}

			//inform user of socket number - used in send and receive commands
			printf("New connection , socket fd is %d , ip is : %s , port : %d\n", fd, inet_ntoa(address.sin_addr), ntohs(address.sin_port));


			save = get_memory(sizeof(t_list_element) + sizeof(t_client));
			client = (t_client*)save->data;
			memset(client, 0, sizeof(t_client));

			client->fd = fd;

			add_list_element(clients, save);
		}

		//Check if there are some IO operation on other sockets
		tmp = clients->head;
		while (tmp != NULL)
		{
			client = (t_client*)tmp->data;
			fd = client->fd;
			if (FD_ISSET(fd, &readfds))
			{
				//Check if it was for closing and read the incoming message
				if ((ret_value = read(fd, &client->buffer[client->buffer_index], BUFFER_SIZE - client->buffer_index)) == 0)
				{
					//A client is disconnected, get his details and print
					getpeername(fd, (struct sockaddr*)&address, (socklen_t*)&addrlen);
					printf("Host disconnected , ip %s , port %d \n", inet_ntoa(address.sin_addr), ntohs(address.sin_port));

					//Close the socket
					close(fd);

					client->fd = -1;
				}
				else
				{
					if (count_message == -1)
					{
						count_message = 0;
						printf("-----------------------------------------------------------------------------\n");
						printf("SERVER:\n");
						before = get_timestamp_microsecond();
					}

					client->buffer_index += ret_value;

					//If there is enough data to get the message's size
					while (client->buffer_index >= (int)sizeof(int))
					{
						message = (t_message*)client->buffer;
						message_size = message->size;

						//if the message_size is impossible (cheater or network error)
						if (message_size < sizeof(t_message) || message_size > BUFFER_SIZE)
							client->buffer_index = 0;
						else if (client->buffer_index >= message_size)
						{
							save = get_memory(sizeof(t_list_element) + message_size);
							message = (t_message*)save->data;

							//Copy datas
							memcpy(message, client->buffer, message_size);

							//Change the size to have only the data size
							message->size = message_size - sizeof(t_message);
							add_list_element(&client->messages, save);

							//Shift datas in the buffer (circular buffer)
							client->buffer_index -= message_size;
							memmove(client->buffer, &client->buffer[message_size], client->buffer_index);

							++count_message;
							if (count_message >= 100000)
							{
								duration = get_timestamp_microsecond() - before;
								printf("100000 messages received in :\t\ttime elapsed = %ld microseconds\n", duration);
								exit(0);
							}
						}
						else
							break;
					}
				}
			}
			tmp = tmp->next;
		}
	}

	return 0;
}

