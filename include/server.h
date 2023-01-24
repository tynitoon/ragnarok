#ifndef SERVER_H
#define SERVER_H

#include <pthread.h>

#include "list.h"

#define BUFFER_SIZE 4096

typedef enum            s_connection_state
{
    DISCONNECTED        = 0,
    CONNECTED           = 1,
    READY_TO_BE_REMOVED = 2
}                       t_connection_state;

typedef struct          s_client
{
    t_connection_state  state;
    int                 fd;
    t_list              messages;
    size_t              buffer_index;
    char                buffer[BUFFER_SIZE];
    pthread_mutex_t     mutex;
}                       t_client;

int start_server(int port, t_list* clients);

#endif
