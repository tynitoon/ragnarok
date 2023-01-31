#include <arpa/inet.h> // inet_addr()
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h> // bzero()
#include <sys/socket.h>
#include <unistd.h> // read(), write(), close()
#define MAX 80
#define PORT 4242
#define SA struct sockaddr

typedef enum    e_data_type
{
    MESSAGE = 0,
}               t_data_type;

typedef struct  s_message
{
	size_t      size;
	t_data_type type;
	char        buffer[];
}               t_message;

void func(int sockfd)
{
  t_message *message;
  int size;
  int i;
  int ret_value;

  srand(time(NULL));
  unsigned long start = time(NULL);
	for (i = 0; i < 10000; ++i) {
	  size = sizeof(t_message) + rand() % 600;
	  message = malloc(size);
	  message->size = size;
	  message->type = MESSAGE;
	  memset(message->buffer, 0, size - sizeof(t_message));
	  
	  //bzero(buff, sizeof(buff));
	  //printf("Enter the string : ");
		//		n = 0;
		//		while ((buff[n++] = getchar()) != '\n')
		//			;
		ret_value = write(sockfd, message, size);
		//if (ret_value == -1)
		//	printf("error\n");
		//else
		//	printf("size = %d ret_value = %d\n", size, ret_value);

		//		if ((strncmp(buff, "exit", 4)) == 0) {
		//			printf("Client Exit...\n");
		//		break;
		//}
		free(message);
		sleep(1);
	}
}

int main()
{
	int sockfd, connfd;
	struct sockaddr_in servaddr, cli;

	// socket create and verification
	sockfd = socket(AF_INET, SOCK_STREAM, 0);
	if (sockfd == -1) {
		printf("socket creation failed...\n");
		exit(0);
	}
	else
		printf("Socket successfully created..\n");
	bzero(&servaddr, sizeof(servaddr));

	// assign IP, PORT
	servaddr.sin_family = AF_INET;
	servaddr.sin_addr.s_addr = inet_addr("127.0.0.1");
	servaddr.sin_port = htons(PORT);

	// connect the client socket to server socket
	if (connect(sockfd, (SA*)&servaddr, sizeof(servaddr))
		!= 0) {
		printf("connection with the server failed...\n");
		exit(0);
	}
	else
		printf("connected to the server..\n");

	// function for chat
	func(sockfd);

	// close the socket
	close(sockfd);
}
