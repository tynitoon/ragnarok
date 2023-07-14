#ifdef linux

#include <stdio.h>
#include <string.h>

#include "map.h"
#include "single_memory.h"

#define FIRST_ALLOCATION (1 << 7) // 128 t_map_element

static void grow_map(t_map* map)
{
	t_map_element*	element;
	t_map_element*	save;
	unsigned long	index;
	unsigned long	i;
	unsigned long	oldsize;

	if (map->size == 0)
	{
		map->datas = (t_map_element**)realloc_memory(map->datas, FIRST_ALLOCATION * sizeof(t_map_element*));
		memset(map->datas, 0, FIRST_ALLOCATION * sizeof(t_map_element*));
		map->size = FIRST_ALLOCATION;
	}
	else
	{
		oldsize = map->size;
		map->size <<= 1;	
		map->datas = (t_map_element**)realloc_memory(map->datas, map->size * sizeof(t_map_element*));
		memset(&map->datas[oldsize], 0, oldsize * sizeof(t_map_element*));

		for (i = 0; i < oldsize; ++i)
		{
			if (map->datas[i] != NULL)
			{
				element = map->datas[i];
				save = NULL;
				while (element != NULL)
				{
					if ((index = element->key % map->size) != i)
					{
						if (save == NULL)
							map->datas[i] = element->next;
						else
							save->next = element->next;

						element->next = map->datas[index];
						map->datas[index] = element;

						if (save == NULL)
							element = map->datas[i];
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

void init_map(t_map* map)
{
	memset(map, 0, sizeof(t_map));
	grow_map(map);
}

void add_map_element(t_map* map, unsigned long key, void* data)
{
	unsigned long	index;
	t_map_element*	element;
	t_map_element*	tmp;

	pthread_mutex_lock(&map->mutex);

	index = key % map->size;

	//Check if already present and try to replace value
	tmp = map->datas[index];
	while (tmp != NULL)
	{
		if (tmp->key == key)
		{
			tmp->data = data;

			pthread_mutex_unlock(&map->mutex);

			return;
		}

		tmp = tmp->next;
	}

	//It's a new element so we create it and link it
	element = (t_map_element*)get_memory(sizeof(t_map_element));
	element->key = key;
	element->data = data;

	element->next = map->datas[index];
	map->datas[index] = element;

	++map->elements;
	if ((double)map->elements / (double)map->size > 0.75)
		grow_map(map);

	pthread_mutex_unlock(&map->mutex);
}

void* get_map_element(t_map* map, unsigned long key)
{
	t_map_element*	tmp;

	pthread_mutex_lock(&map->mutex);

	tmp = map->datas[key % map->size];
	while (tmp != NULL)
	{
		if (tmp->key == key)
		{
			pthread_mutex_unlock(&map->mutex);

			return tmp->data;
		}
		tmp = tmp->next;
	}

	pthread_mutex_unlock(&map->mutex);

	return NULL;
}

void* remove_map_element(t_map* map, unsigned long key)
{
	unsigned long	index;
	t_map_element*	tmp;
	t_map_element*	save = NULL;
	void*			data;

	pthread_mutex_lock(&map->mutex);

	if (map->size == 0)
	{
		pthread_mutex_unlock(&map->mutex);

		return NULL;
	}

	index = key % map->size;

	tmp = map->datas[index];
	while (tmp != NULL)
	{
		if (tmp->key == key)
		{
			if (save == NULL)
				map->datas[index] = tmp->next;
			else
				save->next = tmp->next;

			data = tmp->data;
			free_memory(tmp);
			--map->elements;

			pthread_mutex_unlock(&map->mutex);

			return data;
		}

		save = tmp;
		tmp = tmp->next;
	}

	pthread_mutex_unlock(&map->mutex);

	return NULL;
}

void delete_map(t_map* map)
{
	t_map_element*	element;
	t_map_element*	save;
	unsigned long	i;

	pthread_mutex_lock(&map->mutex);

	for (i = 0; i < map->size; ++i)
	{
		if (map->datas[i] != NULL)
		{
			element = map->datas[i];
			while (element != NULL)
			{
				save = element->next;
				free_memory(element);

				element = save;
			}
		}
	}
	free_memory(map->datas);
	map->elements = 0;
	map->size = 0;

	pthread_mutex_unlock(&map->mutex);
}

void display_map(t_map* map)
{
	t_map_element*	tmp;
	uint64_t		i;
	uint64_t		count_element = 0;

	for (i = 0; i < map->size; ++i)
	{
		tmp = map->datas[i];
		while (tmp != NULL)
		{
			printf("index = %ld, key = %lu, data = %p\n", i, tmp->key, tmp->data);
			++count_element;

			tmp = tmp->next;
		}
	}
	printf("total size map = %ld count element = %ld\n", map->size, count_element);
}

#else

#include <Windows.h>
#include <stdio.h>
#include <string.h>

#include "map.h"
#include "single_memory.h"

#define FIRST_ALLOCATION (1 << 7) // 128 t_map_element

static void init_mutex(t_map* map)
{
	if (map->mutex == NULL)
		map->mutex = CreateMutex(NULL, FALSE, NULL);
}

static void grow_map(t_map* map)
{
	t_map_element* element;
	t_map_element* save;
	unsigned long	index;
	unsigned long	i;
	unsigned long	oldsize;

	if (map->size == 0)
	{
		map->datas = (t_map_element**)realloc_memory(map->datas, FIRST_ALLOCATION * sizeof(t_map_element*));
		memset(map->datas, 0, FIRST_ALLOCATION * sizeof(t_map_element*));
		map->size = FIRST_ALLOCATION;
	}
	else
	{
		oldsize = map->size;
		map->size <<= 1;
		map->datas = (t_map_element**)realloc_memory(map->datas, map->size * sizeof(t_map_element*));
		memset(&map->datas[oldsize], 0, oldsize * sizeof(t_map_element*));

		for (i = 0; i < oldsize; ++i)
		{
			if (map->datas[i] != NULL)
			{
				element = map->datas[i];
				save = NULL;
				while (element != NULL)
				{
					if ((index = element->key % map->size) != i)
					{
						if (save == NULL)
							map->datas[i] = element->next;
						else
							save->next = element->next;

						element->next = map->datas[index];
						map->datas[index] = element;

						if (save == NULL)
							element = map->datas[i];
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

void init_map(t_map* map)
{
	memset(map, 0, sizeof(t_map));
	grow_map(map);
}

void add_map_element(t_map* map, unsigned long key, void* data)
{
	unsigned long	index;
	t_map_element*	element;
	t_map_element*	tmp;

	init_mutex(map);
	WaitForSingleObject(map->mutex, INFINITE);

	index = key % map->size;

	//Check if already present and try to replace value
	tmp = map->datas[index];
	while (tmp != NULL)
	{
		if (tmp->key == key)
		{
			tmp->data = data;

			ReleaseMutex(map->mutex);

			return;
		}

		tmp = tmp->next;
	}

	//It's a new element so we create it and link it
	element = (t_map_element*)get_memory(sizeof(t_map_element));
	element->key = key;
	element->data = data;

	element->next = map->datas[index];
	map->datas[index] = element;

	++map->elements;
	if ((double)map->elements / (double)map->size > 0.75)
		grow_map(map);

	ReleaseMutex(map->mutex);
}

void* get_map_element(t_map* map, unsigned long key)
{
	t_map_element* tmp;

	init_mutex(map);
	WaitForSingleObject(map->mutex, INFINITE);

	tmp = map->datas[key % map->size];
	while (tmp != NULL)
	{
		if (tmp->key == key)
		{
			ReleaseMutex(map->mutex);

			return tmp->data;
		}
		tmp = tmp->next;
	}

	ReleaseMutex(map->mutex);

	return NULL;
}

void* remove_map_element(t_map* map, unsigned long key)
{
	unsigned long	index;
	t_map_element*	tmp;
	t_map_element*	save = NULL;
	void*			data;

	init_mutex(map);
	WaitForSingleObject(map->mutex, INFINITE);

	if (map->size == 0)
	{
		ReleaseMutex(map->mutex);

		return NULL;
	}

	index = key % map->size;

	tmp = map->datas[index];
	while (tmp != NULL)
	{
		if (tmp->key == key)
		{
			if (save == NULL)
				map->datas[index] = tmp->next;
			else
				save->next = tmp->next;

			data = tmp->data;
			free_memory(tmp);
			--map->elements;

			ReleaseMutex(map->mutex);

			return data;
		}

		save = tmp;
		tmp = tmp->next;
	}

	ReleaseMutex(map->mutex);

	return NULL;
}

void delete_map(t_map* map)
{
	t_map_element*	element;
	t_map_element*	save;
	unsigned long	i;

	init_mutex(map);
	WaitForSingleObject(map->mutex, INFINITE);

	for (i = 0; i < map->size; ++i)
	{
		if (map->datas[i] != NULL)
		{
			element = map->datas[i];
			while (element != NULL)
			{
				save = element->next;
				free_memory(element);

				element = save;
			}
		}
	}
	free_memory(map->datas);
	map->elements = 0;
	map->size = 0;

	ReleaseMutex(map->mutex);
}

void display_map(t_map* map)
{
	t_map_element*	tmp;
	uint64_t		i;
	uint64_t		count_element = 0;

	for (i = 0; i < map->size; ++i)
	{
		tmp = map->datas[i];
		while (tmp != NULL)
		{
			printf("index = %ld, key = %lu, data = %p\n", i, tmp->key, tmp->data);
			++count_element;

			tmp = tmp->next;
		}
	}
	printf("total size map = %ld count element = %ld\n", map->size, count_element);
}

#endif