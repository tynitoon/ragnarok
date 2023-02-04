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
#include "single_memory.h"

#define MAX 4096

int main()
{
    int                 fd;
    struct sockaddr_in  servaddr;
    char                buffer[MAX];
    int                 i;
    t_message           message;
    t_connect_message   connect_message;

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
    message.type = CONNECT;
    message.size = sizeof(t_message) + sizeof(t_connect_message);
    printf("message.size = %lu %lu %lu\n", message.size, sizeof(t_message), sizeof(t_connect_message));
    memcpy(connect_message.username, "default\0", strlen("default") + 1);
    memcpy(connect_message.password, "default\0", strlen("default") + 1);

    memcpy(buffer, &message, sizeof(t_message));
    memcpy(&buffer[sizeof(t_message)], &connect_message, sizeof(t_connect_message));

    printf("user = %s password = %s\n", (char*)(&buffer[sizeof(t_message)]), (char*)(&buffer[sizeof(t_message) + 32]));

    if (write(fd, buffer, message.size) < 0)
        return -1;

    // close the socket
    close(fd);

    return 0;
}