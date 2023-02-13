#ifndef CLIENT_H
#define CLIENT_H

#include <stdint.h>
#include "protocol.h"
#include "list.h"

#define BUFFER_SIZE 4096

typedef struct  s_server
{
    SOCKET      fd;
    t_list      messages;
    uint64_t    buffer_index;
    char        buffer[BUFFER_SIZE];
}               t_server;

int start_client(char* address, char* port, t_server* server);

#endif
