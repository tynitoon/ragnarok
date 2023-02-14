#ifndef LIST_H
#define LIST_H

#ifdef linux

#include <pthread.h>

typedef struct              s_list_element
{
    struct s_list_element*  prev;
    struct s_list_element*  next;
    int                     reading_thread;
    char                    data[];
}                           t_list_element;

typedef struct              s_list
{
    pthread_mutex_t         mutex;
    t_list_element*         head;
    t_list_element*         tail;
}                           t_list;

void            add_list_element(t_list* list, t_list_element* to_add);
t_list_element* remove_list_element(t_list* list, t_list_element* to_remove);
void            display_list(t_list* list);

#else

/* Define Things to avoid include windows.h */
typedef void*               HANDLE;
/* End of defines for windows.h */

typedef struct              s_list_element
{
    struct s_list_element*  prev;
    struct s_list_element*  next;
    int                     reading_thread;
    char                    data[];
}                           t_list_element;

typedef struct              s_list
{
    HANDLE                  mutex;
    t_list_element*         head;
    t_list_element*         tail;
}                           t_list;

void            add_list_element(t_list* list, t_list_element* to_add);
t_list_element* remove_list_element(t_list* list, t_list_element* to_remove);
void            display_list(t_list* list);

#endif

#endif