#include <string.h>
#include <stdio.h>

#include "single_memory.h"
#include "list.h"

void add_list_element(t_list* list, t_list_element* to_add)
{
	pthread_mutex_lock(&list->mutex);

	to_add->next = NULL;
	to_add->prev = list->tail;

	if (list->tail == NULL)
	{
		list->tail = to_add;
		list->head = to_add;
	}
	else
	{
		list->tail->next = to_add;
		list->tail = to_add;
	}

	pthread_mutex_unlock(&list->mutex);
}

t_list_element* remove_list_element(t_list* list, t_list_element* to_remove)
{
	if (to_remove == NULL)
		return NULL;

	pthread_mutex_lock(&list->mutex);

	if (list->head == to_remove)
	{
		list->head = to_remove->next;
		if (list->head != NULL)
			list->head->prev = NULL;
		else
			list->tail = NULL;
	}
	else if (list->tail == to_remove)
	{
		list->tail = to_remove->prev;
		if (list->tail != NULL)
			list->tail->next = NULL;
		else
			list->head = NULL;
	}
	else
	{
		to_remove->prev->next = to_remove->next;
		to_remove->next->prev = to_remove->prev;
	}

	pthread_mutex_unlock(&list->mutex);
	
	return to_remove;
}

void display_list(t_list* list)
{
	t_list_element* element;
	int				count = 0;

	printf("\nDisplay list :\n");
	element = list->head;
	while (element != NULL)
	{
		printf("element = %ld\n", (unsigned long)element);
		element = element->next;
		++count;
	}
	printf("List size = %d\n", count);

}