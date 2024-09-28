#ifndef MAP_H
#define MAP_H

#include <stddef.h>

#include "mutex"

typedef struct				s_map_element
{
	unsigned long			key;
	void*					data;
	struct s_map_element*	next;
}							t_map_element;

typedef struct				s_map
{
	MUTEX					mutex;
	t_map_element**			datas;
	size_t					size;
	size_t					elements;
}							t_map;

/*
 * /brief initialize a map
 *
 * /param[in] map is the map to initialize
 */
void init_map(t_map* map);

/*
 * /brief add element in map, if the key already exists, the data is replaced
 *
 * /param[in] map is the map where the element will be added
 * /param[in] key is an id, you can use hash.h to generate your own key or increment an unsigned long
 * /param[in] data is the data to store
 */
void add_map_element(t_map* map, unsigned long key, void* data);

/*
 * /brief retrieve data from a key
 *
 * /param[in] map is the map where elements are stored
 * /param[in] key is an id corresponding to a data
 *
 * /return the data or NULL if the key doesn't exist
 */
void* get_map_element(t_map* map, unsigned long key);

/*
 * /brief remove data from map
 *
 * /param[in] map is the map where elements are stored
 * /param[in] key is an id corresponding to a data
 *
 * /return the removed data or NULL if the key doesn't exist
 */
void* remove_map_element(t_map* map, unsigned long key);

/*
 * /brief delete a map, you have to init it again if you want to reuse it
 *
 * /param[in] map is the map where elements are stored
 */
void delete_map(t_map* map);

/*
 * /brief display all index, key and data. It also displays the used size and the total size of the map
 *
 * /param[in] map is the map where elements are stored
 */
void display_map(t_map* map);

#endif /* MAP_H */