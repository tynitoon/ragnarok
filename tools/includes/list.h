#ifndef LIST_H
#define LIST_H

#ifdef linux

#include <pthread.h>

typedef struct             s_list_element
{
	struct s_list_element* prev;
	struct s_list_element* next;
	int                    reading_thread;
	char                   data[];
}                          t_list_element;

typedef struct             s_list
{
	pthread_mutex_t        mutex;
	t_list_element*        head;
	t_list_element*        tail;
}                          t_list;

void            add_list_element(t_list* list, t_list_element* to_add);
t_list_element* remove_list_element(t_list* list, t_list_element* to_remove);
void            display_list(t_list* list);

#else

/* Define Things to avoid include windows.h */
typedef void*              HANDLE;
/* End of defines for windows.h */

/*
 * /brief list element of a double linked list
 */
typedef struct             s_list_element
{
	struct s_list_element* prev;
	struct s_list_element* next;
	char                   data[];
}                          t_list_element;

/*
 * /brief double linked list
 */
typedef struct              s_list
{
	HANDLE                  mutex;
	t_list_element*         head;
	t_list_element*         tail;
}                           t_list;

/*
 * /brief add an element at the end of the list
 * 
 * /param[in] list structure, it has to be initialized before
 * /param[in] to_add is the element to add (Must not be NULL)
 */
void            add_list_element(t_list* list, t_list_element* to_add);

/*
 * /brief add an element at the end of the list
 *
 * /param[in] list structure, it has to be initialized before
 * /param[in] to_remove is the element to remove
 *
 * /note there is no check that the element is in the list
 */
t_list_element* remove_list_element(t_list* list, t_list_element* to_remove);

/*
 * /brief display all elements addresses
 */
void            display_list(t_list* list);

#endif

#endif