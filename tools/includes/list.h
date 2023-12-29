#ifndef LIST_H
#define LIST_H

/* Definitions to avoid duplication code for crossplatform
   DO NOT INCLUDE <windows.h> here or raylib will not be able to compile */
#ifdef linux
#include <pthread.h>

typedef pthread_mutex_t		MUTEX;
#else
typedef void*				MUTEX;
#endif /* Linux or windows */

typedef struct				s_list_element
{
	struct s_list_element*	prev;
	struct s_list_element*	next;
	char					data[];
}							t_list_element;

typedef struct				s_list
{
	MUTEX					mutex;
	t_list_element*			head;
	t_list_element*			tail;
}							t_list;

/*
 * /brief initialize a list
 *
 * /param[in] list is the list to initialize
 */
void init_list(t_list* list);

/*
 * /brief add an element at the end of the list
 * 
 * /param[in] list structure, it has to be initialized before
 * /param[in] to_add is the element to add (Must not be NULL)
 */
void add_list_element(t_list* list, t_list_element* to_add);

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

#endif /* LIST_H */