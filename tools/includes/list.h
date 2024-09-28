#ifndef LIST_H
#define LIST_H

#include "mutex.h"

typedef struct				s_list_element
{
	struct s_list_element*	prev;
	struct s_list_element*	next;
	void*					data;
}							t_list_element;

typedef struct
{
	MUTEX			mutex;
	t_list_element*	head;
	t_list_element*	tail;
	size_t			nb_elements;
}					t_list;

/*
 * /brief initialize a list
 *
 * /param[in] list is the list to initialize
 */
void list_init(t_list* list);

/*
 * /brief store the data at the front of the list
 * 
 * /param[in] list structure, it has to be initialized before
 * /param[in] to_add is the data to store in the list (NULL will be ignored)
 */
void list_add_front(t_list* list, void* to_add);

/*
 * /brief store the data at the back of the list
 *
 * /param[in] list structure, it has to be initialized before
 * /param[in] to_add is the data to store in the list (NULL will be ignored)
 */
void list_add_back(t_list* list, void* to_add);

/*
 * /brief store the data at the asked position in the list
 *
 * /param[in] list structure, it has to be initialized before
 * /param[in] to_add is the data to store in the list (NULL will be ignored)
 * /param[in] position is the position where the data will be stored, if it's higher than the number of elements of the list, then it will be put at the tail of the list
 */
void list_add_at(t_list* list, void* to_add, size_t position);

/*
 * /brief remove the data at the front of the list and returns it
 *
 * /param[in] list structure, it has to be initialized before
 * 
 * /return the stored data 
 */
void* list_remove_front(t_list* list);

/*
 * /brief remove the data at the back of the list and returns it
 *
 * /param[in] list structure, it has to be initialized before
 *
 * /return the stored data
 */
void* list_remove_back(t_list* list);

/*
 * /brief remove the data at the position of the list and returns it
 *
 * /param[in] list structure, it has to be initialized before
 * /param[in] position is the position of the data to remove
 *
 * /return the stored data
 */
void* list_remove_at(t_list* list, size_t position);

/*
 * /brief search and remove the data from the list and returns it (it stops at the first match)
 *
 * /param[in] list structure, it has to be initialized before
 *
 * /return the stored data
 */
void* list_remove(t_list* list, void* to_remove);

/*
 * /brief remove every element of the list (doesn't not free the data so you need to have them somewhere else)
 *
 * /param[in] list structure, it has to be initialized before
 */
void list_clear(t_list* list);

/*
 * /brief search the data in the list and returns its position (it stops at the first match)
 *
 * /param[in] list structure, it has to be initialized before
 * /param[in] to_find is the data to find
 *
 * /return the position of the first match
 */
int list_find(t_list* list, void* to_find);

/*
 * /brief display all elements addresses
 */
void list_display(t_list* list);

#endif /* LIST_H */