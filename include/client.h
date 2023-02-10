#ifndef CLIENT_H
#define CLIENT_H

#include "list.h"

#define BUFFER_SIZE 4096

typedef struct  s_server
{
    int         fd;
    t_list      messages;
    size_t      buffer_index;
    char        buffer[BUFFER_SIZE];
}               t_server;

int start_client(char* address, int port, t_server* server);

#endif
