#ifdef _WIN32
#include <Windows.h>
#endif

#include <stdint.h>
#include <string.h>
#include <stdio.h>

#include "single_memory.h"
#include "list.h"

void list_init(t_list* list)
{
	memset(list, 0, sizeof(t_list));
	mutex_init(&list->mutex);
}

void list_destroy(t_list* list)
{
	list_clear(list);
	mutex_destroy(&list->mutex);
}

void list_add_front(t_list* list, void* to_add)
{
	if (to_add == NULL)
	{
		fprintf(stderr, "list_add_front: NULL data has been ignored\n");
		return;
	}

	mutex_lock(&list->mutex);

	//Init element
	t_list_element* element = MALLOC(sizeof(t_list_element));
	element->data = to_add;
	element->prev = NULL;
	element->next = list->head;

	//If the list is empty head == tail == element, otherwise the new element replace the head
	if (list->nb_elements == 0)
	{
		list->head = element;
		list->tail = element;
	}
	else
	{
		list->head->prev = element;
		list->head = element;
	}

	++list->nb_elements;

	mutex_unlock(&list->mutex);
}

void list_add_back(t_list* list, void* to_add)
{
	if (to_add == NULL)
	{
		fprintf(stderr, "list_add_back: NULL data has been ignored\n");
		return;
	}

	mutex_lock(&list->mutex);

	//Init the element
	t_list_element* element = MALLOC(sizeof(t_list_element));
	element->data = to_add;
	element->next = NULL;
	element->prev = list->tail;

	//If the list is empty head == tail == element, otherwise the new element replace the tail
	if (list->nb_elements == 0)
	{
		list->head = element;
		list->tail = element;
	}
	else
	{
		list->tail->next = element;
		list->tail = element;
	}

	++list->nb_elements;

	mutex_unlock(&list->mutex);
}

void list_add_at(t_list* list, void* to_add, size_t position)
{
	if (to_add == NULL)
	{
		fprintf(stderr, "list_add_back: NULL data has been ignored\n");
		return;
	}

	mutex_lock(&list->mutex);

	t_list_element* element = MALLOC(sizeof(t_list_element));
	element->data = to_add;
	if (list->nb_elements == 0 || position == 0)
	{
		//Copy of list_add_front
		element->prev = NULL;
		element->next = list->head;
		if (list->nb_elements == 0)
		{
			list->head = element;
			list->tail = element;
		}
		else
		{
			list->head->prev = element;
			list->head = element;
		}
	}
	else if (position >= list->nb_elements)
	{
		//Copy of list_add_back
		element->next = NULL;
		element->prev = list->tail;
		list->tail->next = element;
		list->tail = element;
	}
	else
	{
		//Add at position
		t_list_element* tmp;
		size_t currentPosition;

		//A little optimisation
		if (position < list->nb_elements / 2)
		{
			currentPosition = 0;
			tmp = list->head;
			while (currentPosition != position)
			{
				tmp = tmp->next;
				++currentPosition;
			}
		}
		else
		{
			currentPosition = list->nb_elements - 1;
			tmp = list->tail;
			while (currentPosition != position)
			{
				tmp = tmp->prev;
				--currentPosition;
			}
		}

		//Insert the element
		element->next = tmp;
		element->prev = tmp->prev;
		tmp->prev = element;
		if (element->prev != NULL)
			element->prev->next = element;
	}

	++list->nb_elements;

	mutex_unlock(&list->mutex);
}

void* list_remove_front(t_list* list)
{
	void* data;

	mutex_lock(&list->mutex);

	if (list->nb_elements == 0)
		data = NULL;
	else
	{
		//Store the front element
		t_list_element* front_element = list->head;

		//Update the list
		list->head = list->head->next;
		if (list->head == NULL)
			list->tail = NULL;
		else
			list->head->prev = NULL;
		--list->nb_elements;

		//Store the data and free the container
		data = front_element->data;
		FREE(front_element);
	}

	mutex_unlock(&list->mutex);

	return data;
}

void* list_remove_back(t_list* list)
{
	void* data;

	mutex_lock(&list->mutex);

	if (list->nb_elements == 0)
		data = NULL;
	else
	{
		//Get the back element and free the container
		t_list_element* back_element = list->tail;

		//Update the list
		list->tail = list->tail->prev;
		if (list->tail == NULL)
			list->head = NULL;
		else
			list->tail->next = NULL;
		--list->nb_elements;

		//Store the data and free the container
		data = back_element->data;
		FREE(back_element);
	}

	mutex_unlock(&list->mutex);

	return data;
}

void* list_remove_at(t_list* list, size_t position)
{
	void* data;
	t_list_element* removed_element = NULL;

	mutex_lock(&list->mutex);

	if (list->nb_elements == 0 || position > list->nb_elements - 1)
		data = NULL;
	else if (position == 0)
	{
		//Copy of list_remove_front
		removed_element = list->head;

		list->head = list->head->next;
		if (list->head == NULL)
			list->tail = NULL;
		else
			list->head->prev = NULL;
	}
	else if (position == list->nb_elements - 1)
	{
		//Copy of list_remove_back
		removed_element = list->tail;

		list->tail = list->tail->prev;
		if (list->tail == NULL)
			list->head = NULL;
		else
			list->tail->next = NULL;
	}
	else
	{
		//Remove at position
		size_t currentPosition;

		//A little optimisation
		if (position < list->nb_elements / 2)
		{
			currentPosition = 0;
			removed_element = list->head;
			while (currentPosition != position)
			{
				removed_element = removed_element->next;
				++currentPosition;
			}
		}
		else
		{
			currentPosition = list->nb_elements - 1;
			removed_element = list->tail;
			while (currentPosition != position)
			{
				removed_element = removed_element->prev;
				--currentPosition;
			}
		}

		//Link previous and next element between them
		removed_element->prev->next = removed_element->next;
		removed_element->next->prev = removed_element->prev;
	}

	if (removed_element != NULL)
	{
		--list->nb_elements;
		data = removed_element->data;
		FREE(removed_element);
	}

	mutex_unlock(&list->mutex);

	return data;
}


void* list_remove(t_list* list, void* to_remove)
{
	void* data;

	mutex_lock(&list->mutex);

	if (list->nb_elements == 0)
		data = NULL;
	else
	{
		t_list_element* removed_element = list->head;
		while (removed_element->data != to_remove && removed_element->next != NULL)
			removed_element = removed_element->next;

		if (removed_element->data == to_remove)
		{
			if (removed_element == list->head)
				list->head = removed_element->next;
			if (removed_element == list->tail)
				list->tail = removed_element->prev;

			if (removed_element->prev != NULL)
				removed_element->prev->next = removed_element->next;
			if (removed_element->next != NULL)
				removed_element->next->prev = removed_element->prev;

			FREE(removed_element);

			data = to_remove;
			--list->nb_elements;
		}
		else
			data = NULL;
	}

	mutex_unlock(&list->mutex);

	return data;
}

void list_clear(t_list* list)
{
	mutex_lock(&list->mutex);

	t_list_element* save;
	t_list_element* element = list->head;
	while (element != NULL)
	{
		save = element;
		element = element->next;
		FREE(save);
	}

	list->head = NULL;
	list->tail = NULL;
	list->nb_elements = 0;

	mutex_unlock(&list->mutex);
}

int list_find(t_list* list, void* to_find)
{
	int position = 0;

	mutex_lock(&list->mutex);

	if (list->nb_elements == 0)
		position = -1;
	else
	{
		t_list_element* element = list->head;
		while (element->data != to_find && element->next != NULL)
		{
			element = element->next;
			++position;
		}

		if (element->data != to_find)
			position = -1;
	}

	mutex_unlock(&list->mutex);

	return position;
}


void list_display(t_list* list)
{
	mutex_lock(&list->mutex);

	printf("Display list:\n");
	printf("head = %ld tail = %ld nb of elements = %ld\n", (uint64_t)list->head, (uint64_t)list->tail, list->nb_elements);
	printf("All elements:\n");
	t_list_element* element = list->head;
	while (element != NULL)
	{
		printf("element = %ld element->prev = %ld element->next = %ld element->data = %ld\n", (uint64_t)element, (uint64_t)element->prev, (uint64_t)element->next, (uint64_t)element->data);
		element = element->next;
	}

	mutex_unlock(&list->mutex);
}