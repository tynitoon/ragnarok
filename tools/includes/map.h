#ifndef MAP_H_
#define MAP_H_

#ifdef linux

#include <pthread.h>
#include <stddef.h>

typedef struct              s_map_element
{
    unsigned long           key;
    void*                   data;
    struct s_map_element*   next;
}                           t_map_element;

typedef struct              s_map
{
    pthread_mutex_t         mutex;
    t_map_element**         datas;
    size_t                  size;
    size_t                  elements;
}                           t_map;

void    init_map(t_map* map);
void    add_map_element(t_map* map,unsigned long key, void* data);
void*   get_map_element(t_map* map, unsigned long key);
void*   remove_map_element(t_map* map, unsigned long key);
void    delete_map(t_map* map);
void    display_map(t_map* map);

#else

#include <stddef.h>

/* Define Things to avoid include windows.h */
typedef void* HANDLE;
/* End of defines for windows.h */

typedef struct              s_map_element
{
    unsigned long           key;
    void* data;
    struct s_map_element* next;
}                           t_map_element;

typedef struct              s_map
{
    HANDLE                  mutex;
    t_map_element**         datas;
    size_t                  size;
    size_t                  elements;
}                           t_map;

void    init_map(t_map* map);
void    add_map_element(t_map* map, unsigned long key, void* data);
void*   get_map_element(t_map* map, unsigned long key);
void*   remove_map_element(t_map* map, unsigned long key);
void    delete_map(t_map* map);
void    display_map(t_map* map);

#endif

#endif