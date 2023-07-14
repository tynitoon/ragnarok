#include <arpa/inet.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <unistd.h>
#include <time.h>

#include "protocol.h"

#define MAX 1000

int main()
{
	int                fd;
	struct sockaddr_in servaddr;
	char               buffer[MAX];
	int                i;
	t_message          message;

	// socket create and verification
	fd = socket(AF_INET, SOCK_STREAM, 0);
	if (fd == -1) {
		printf("socket creation failed...\n");
		return 1;
	}

	memset(&servaddr, 0, sizeof(servaddr));

	// assign IP, PORT
	servaddr.sin_family = AF_INET;
	servaddr.sin_addr.s_addr = inet_addr("127.0.0.1");
	servaddr.sin_port = htons(4242);

	// connect the client socket to server socket
	if (connect(fd, (struct sockaddr*)&servaddr, sizeof(servaddr)) != 0)
	{
		printf("connection with the server failed...\n");
		return 1;
	}

	// function for chat
	srand(time(NULL));
	memset(&message, 0, sizeof(t_message));
	memset(&buffer, 1, sizeof(buffer));
	for (i = 0; i < 100000; ++i)
	{
		message.type = CONNECT;
		message.size = sizeof(t_message) + rand() % (MAX - sizeof(t_message));
		memcpy(buffer, &message, sizeof(t_message));

		if (write(fd, buffer, message.size) < 0)
			return 0;
	}

	// close the socket
	close(fd);

	return 0;
}