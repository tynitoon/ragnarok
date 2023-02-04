#ifndef SERVER_H
#define SERVER_H

#include <pthread.h>

#include "list.h"

#define BUFFER_SIZE 4096

typedef struct          s_client
{
    int                 fd;
    t_list              messages;
    size_t              buffer_index;
    char                buffer[BUFFER_SIZE];
    pthread_mutex_t     mutex;
}                       t_client;

int start_server(int port, t_list* clients);

#endif
