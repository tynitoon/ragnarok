#ifndef MAP_H_
#define MAP_H_

#include <pthread.h>
#include <stddef.h>

typedef struct              s_map_element
{
    void*                   key;
    void*                   data;
    struct s_map_element*   next;
}                           t_map_element;

typedef struct              s_map
{
    pthread_mutex_t         mutex;
    t_map_element**         datas;
    size_t                  elements;
    size_t                  key_size;
    size_t                  size;
}                           t_map;

void    init_map(t_map* map, size_t key_size);
void    add_map_element(t_map* map, void* key, void* data);
void*   get_map_element(t_map* map, void* key);
void*   remove_map_element(t_map* map, void* key);
void    display_map(t_map* map);

#endif