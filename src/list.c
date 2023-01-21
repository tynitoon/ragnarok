#include <stddef.h>
#include <string.h>
#include <stdio.h>

#include "single_memory.h"
#include "list.h"

void add_to_list(t_list* list, char* buffer, size_t size)
{
	t_list_element* new_element;

	new_element = get_memory(sizeof(t_list_element) + size);
	memcpy(new_element->buffer, buffer, size);

	add_list_element_to_list(list, new_element);

}

void add_list_element_to_list(t_list* list, t_list_element* to_add)
{
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
}

void remove_from_list(t_list* list, t_list_element* to_remove)
{
	if (to_remove == NULL)
		return;

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
}

void display_list(t_list* list)
{
	t_list_element* element;

	printf("\nDisplay list :\n");
	element = list->head;
	while (element != NULL)
	{
		printf("element = %ld\n", (unsigned long)element);
		element = element->next;
	}
}