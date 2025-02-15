#ifndef MAP_H
#define MAP_H

#include <stddef.h>

#include "mutex.h"

/*!
 * \brief map element
 */
typedef struct				s_map_element
{
	unsigned int			key;	/*!< key that is used to retrieve the associated data */
	void*					data;	/*!< store data */
	struct s_map_element*	next;	/*!< next map element */
}							t_map_element;

/*!
 * \brief map container
 */
typedef struct
{
	MUTEX					mutex;			/*!< used to have a threadsafe map */
	t_map_element**			data;			/*!< array of t_map_element* (used to store data) */
	size_t					capacity;		/*!< current max capacity of the map */
	size_t					nb_elements;	/*!< number of element */
}							t_map;

/*!
 * /brief initialize a map
 *
 * /param[in] map is the map to initialize
 */
void map_init(t_map* map);

/*!
 * /brief add element in map, if the key already exists, the data is replaced
 *
 * /param[in] map is the map where the element will be added
 * /param[in] key is an id, you can use hash.h to generate your own key or increment an unsigned long
 * /param[in] data is the data to store
 */
void map_add(t_map* map, unsigned int key, void* data);

/*!
 * /brief retrieve data from a key
 *
 * /param[in] map is the map where elements are stored
 * /param[in] key is an id corresponding to a data
 *
 * /return the data or NULL if the key doesn't exist
 */
void* map_get(t_map* map, unsigned int key);

/*!
 * /brief remove data from map
 *
 * /param[in] map is the map where elements are stored
 * /param[in] key is an id corresponding to a data
 *
 * /return the removed data or NULL if the key doesn't exist
 */
void* map_remove(t_map* map, unsigned int key);

/*!
 * /brief remove all elements of the map and reallocate data like a first call
 *
 * /param[in] map is the map where elements are stored
 */
void map_clear(t_map* map);

/*!
 * /brief delete a map, you will have to init it again if you want to reuse it
 *
 * /param[in] map is the map where elements are stored
 * /remark since it also destroys the mutex, it cannot be threadsafe 
 */
void map_delete(t_map* map);

/*!
 * /brief display all index, key and data. It also displays the used size and the total size of the map
 *
 * /param[in] map is the map where elements are stored
 */
void map_display(t_map* map);

#endif /* MAP_H */