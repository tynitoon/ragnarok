#ifndef LIST_H
#define LIST_H

#include <pthread.h>
#include <stddef.h>

typedef struct              s_list_element2
{
    struct s_list_element* prev;
    struct s_list_element* next;
    char*                    buffer;
}                           t_list_element2;

typedef struct              s_list_element
{
    struct s_list_element*  prev;
    struct s_list_element*  next;
    char                    buffer[];
}                           t_list_element;

typedef struct              s_list
{
    pthread_mutex_t         mutex;
    t_list_element*         head;
    t_list_element*         tail;
}                           t_list;

int             init_list(t_list* list);
void            add_list_element(t_list* list, t_list_element* to_add);
t_list_element* remove_list_element(t_list* list, t_list_element* to_remove);
void            display_list(t_list* list);

#endif