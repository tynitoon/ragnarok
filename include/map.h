#ifndef MAP_H_
#define MAP_H_

#include <pthread.h>
#include <stddef.h>

#include "single_memory.h"
#include "list.h"
#include "hash.h"

typedef struct              s_map
{
    pthread_mutex_t         mutex;
    t_list                  *datas;
}                           t_map;

int     init_map(t_map* map);
void*   get_map_element(t_map* map, const char* key, size_t size);
void    add_map_element(t_map* map, const char* key, size_t size, void* );
//void*   remove_map_element(t_map* map, int key)
void    display_map(t_map* map);

#endif