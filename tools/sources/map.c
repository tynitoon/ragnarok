#include <stdio.h>
#include <string.h>
#include <inttypes.h> //For PRIu64

#include "single_memory.h"
#include "map.h"

#define FIRST_ALLOCATION (1 << 7) //128 t_map_element

static void map_grow(t_map* map)
{
	if (map->size == 0)
	{
		map->data = (t_map_element**)realloc_memory(map->data, FIRST_ALLOCATION * sizeof(t_map_element*));
		memset(map->data, 0, FIRST_ALLOCATION * sizeof(t_map_element*));
		map->size = FIRST_ALLOCATION;
	}
	else
	{
		size_t oldsize = map->size;
		map->size <<= 1;
		map->data = (t_map_element**)realloc_memory(map->data, map->size * sizeof(t_map_element*));
		memset(&map->data[oldsize], 0, oldsize * sizeof(t_map_element*));

		for (size_t i = 0; i < oldsize; ++i)
		{
			if (map->data[i] != NULL)
			{
				t_map_element* element = map->data[i];
				t_map_element* save = NULL;
				while (element != NULL)
				{
					unsigned int index = element->key % map->size;
					if (index != i)
					{
						if (save == NULL)
							map->data[i] = element->next;
						else
							save->next = element->next;

						element->next = map->data[index];
						map->data[index] = element;

						if (save == NULL)
							element = map->data[i];
						else
							element = save->next;
					}
					else
					{
						save = element;
						element = element->next;
					}

				}
			}
		}
	}
}

void map_init(t_map* map)
{
	memset(map, 0, sizeof(t_map));
	mutex_init(&map->mutex);
	map_grow(map);
}

void map_add(t_map* map, unsigned int key, void* data)
{
	mutex_lock(&map->mutex);

	//Check if already present and try to replace value
	unsigned int index = key % map->size;
	t_map_element* tmp = map->data[index];
	while (tmp != NULL)
	{
		if (tmp->key == key)
		{
			tmp->data = data;

			mutex_unlock(&map->mutex);

			return;
		}

		tmp = tmp->next;
	}

	//It's a new element so we create it and link it
	t_map_element* element = (t_map_element*)get_memory(sizeof(t_map_element));
	element->key = key;
	element->data = data;

	element->next = map->data[index];
	map->data[index] = element;

	++map->elements;
	if ((double)map->elements / (double)map->size > 0.75)
		map_grow(map);

	mutex_unlock(&map->mutex);
}

void* map_get(t_map* map, unsigned int key)
{
	mutex_lock(&map->mutex);

	t_map_element* tmp = map->data[key % map->size];
	while (tmp != NULL)
	{
		if (tmp->key == key)
		{
			mutex_unlock(&map->mutex);

			return tmp->data;
		}
		tmp = tmp->next;
	}

	mutex_unlock(&map->mutex);

	return NULL;
}

void* map_remove(t_map* map, unsigned int key)
{
	mutex_lock(&map->mutex);

	if (map->size == 0)
	{
		mutex_unlock(&map->mutex);

		return NULL;
	}

	unsigned int index = key % map->size;

	t_map_element* save = NULL;
	t_map_element* tmp = map->data[index];
	while (tmp != NULL)
	{
		if (tmp->key == key)
		{
			if (save == NULL)
				map->data[index] = tmp->next;
			else
				save->next = tmp->next;

			void* data = tmp->data;
			free_memory(tmp);
			--map->elements;

			mutex_unlock(&map->mutex);

			return data;
		}

		save = tmp;
		tmp = tmp->next;
	}

	mutex_unlock(&map->mutex);

	return NULL;
}

void map_clear(t_map* map)
{
	mutex_lock(&map->mutex);

	//Remove all elements
	for (size_t i = 0; i < map->size; ++i)
	{
		if (map->data[i] != NULL)
		{
			t_map_element* save;
			t_map_element* element = map->data[i];
			while (element != NULL)
			{
				save = element->next;
				free_memory(element);

				element = save;
			}
		}
	}
	map->elements = 0;
	map->size = 0;

	//Reallocate the size to FIRST_ALLOCATION
	map_grow(map);

	mutex_unlock(&map->mutex);
}

void map_delete(t_map* map)
{
	for (size_t i = 0; i < map->size; ++i)
	{
		if (map->data[i] != NULL)
		{
			t_map_element* save;
			t_map_element* element = map->data[i];
			while (element != NULL)
			{
				save = element->next;
				free_memory(element);

				element = save;
			}
		}
	}
	free_memory(map->data);
	map->elements = 0;
	map->size = 0;
}

void map_display(t_map* map)
{
	mutex_lock(&map->mutex);

	size_t count_element = 0;
	for (size_t i = 0; i < map->size; ++i)
	{
		t_map_element* tmp = map->data[i];
		while (tmp != NULL)
		{
			printf("index = %" PRIu64 ", key = %u, data = %p\n", i, tmp->key, tmp->data);
			++count_element;

			tmp = tmp->next;
		}
	}
	printf("total size map = %" PRIu64 " count element = %" PRIu64 "\n", map->size, count_element);

	mutex_unlock(&map->mutex);
}