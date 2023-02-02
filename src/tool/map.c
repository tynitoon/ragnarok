#include <stdio.h>
#include <string.h>

#include "map.h"
#include "single_memory.h"
#include "hash.h"

#define CHUNK_SIZE (1 << 6) // 64 t_map_element

static void refresh_map(t_map* map)
{
	unsigned int	hash;
	t_map_element*	tmp = NULL;
	t_map_element*	tmp2;
	t_map_element*	next;
	t_map_element*	save;
	size_t			i;

	for (i = 0; i < map->size - CHUNK_SIZE; ++i)
	{
		if (map->datas[i] == NULL)
			continue;

		if (tmp == NULL)
		{
			tmp = map->datas[i];
			save = tmp;
		}
		else
			tmp->next = map->datas[i];

		while (tmp->next != NULL)
			tmp = tmp->next;

		map->datas[i] = NULL;
	}

	tmp = save;
	while (tmp != NULL)
	{
		hash = generate_hash(tmp->key, map->key_size);

		next = tmp->next;
		tmp->next = NULL;

		if (map->datas[hash % map->size] == NULL)
			map->datas[hash % map->size] = tmp;
		else
		{
			tmp2 = map->datas[hash % map->size];
			while (tmp2->next != NULL)
				tmp2 = tmp2->next;

			tmp2->next = tmp;
		}

		tmp = next;
	}
}

static void grow_map(t_map* map)
{
	map->datas = realloc_memory(map->datas, (map->size + CHUNK_SIZE) * sizeof(t_map_element*));
	memset(&map->datas[map->size], 0, CHUNK_SIZE * sizeof(t_map_element*));
	map->size += CHUNK_SIZE;
}

void init_map(t_map* map, size_t key_size)
{
	memset(map, 0, sizeof(t_map));
	map->key_size = key_size;
}

void add_map_element(t_map* map, void* key, void* data)
{
	unsigned int	hash;
	t_map_element*	element;
	t_map_element*	tmp;

	pthread_mutex_lock(&map->mutex);

	if (map->size == 0)
		grow_map(map);

	hash = generate_hash(key, map->key_size);

	tmp = map->datas[hash % map->size];
	while (tmp != NULL)
	{
		if (tmp->key == key)
		{
			pthread_mutex_unlock(&map->mutex);

			return;
		}

		if (tmp->next == NULL)
			break;

		tmp = tmp->next;
	}

	element = get_memory(sizeof(t_map_element));
	element->key = key;
	element->data = data;
	element->next = NULL;

	if (tmp == NULL)
		map->datas[hash % map->size] = element;
	else
		tmp->next = element;

	++map->elements;
	if ((double)map->elements / (double)map->size > 0.75)
	{
		grow_map(map);
		refresh_map(map);
	}

	pthread_mutex_unlock(&map->mutex);
}

void* get_map_element(t_map* map, void* key)
{
	unsigned int	hash;
	t_map_element*	tmp;

	pthread_mutex_lock(&map->mutex);

	if (map->size == 0)
	{
		pthread_mutex_unlock(&map->mutex);

		return NULL;
	}

	hash = generate_hash(key, map->key_size);

	tmp = map->datas[hash % map->size];
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

void* remove_map_element(t_map* map, void* key)
{
	unsigned int	hash;
	t_map_element*	tmp;
	t_map_element*	save;
	void*			data;

	pthread_mutex_lock(&map->mutex);

	if (map->size == 0)
	{
		pthread_mutex_unlock(&map->mutex);

		return NULL;
	}

	hash = generate_hash(key, map->key_size);

	tmp = map->datas[hash % map->size];
	while (tmp != NULL)
	{
		if (tmp->key == key)
		{
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

void display_map(t_map* map)
{
	t_map_element*	tmp;
	size_t			i;
	size_t			count_element = 0;

	for (i = 0; i < map->size; ++i)
	{
		tmp = map->datas[i];
		while (tmp != NULL)
		{
			printf("index = %ld, key = %p, key size = %ld, data = %p, hash = %u\n", i, tmp->key, map->key_size, tmp->data, generate_hash(tmp->key, map->key_size));
			++count_element;

			tmp = tmp->next;
		}
	}
	printf("size map = %ld count element = %ld\n", map->size, count_element);
}

//#include <stdio.h>
//
//#include "map.h"
//
//static t_map_element* left_rotation(t_map_element* element)
//{
//	t_map_element*	tmp;
//	int				weight_left = 0;
//	int				weight_right = 0;
//
//	tmp = element;
//	element = element->right;
//	tmp->right = element->left;
//	element->left = tmp;
//	element->weight = tmp->weight;
//
//	if (tmp->left != NULL)
//		weight_left = tmp->left->weight;
//	if (tmp->right != NULL)
//		weight_right = tmp->right->weight;
//
//	tmp->weight = weight_left + weight_right + 1;
//
//	return element;
//}
//
//static t_map_element* right_rotation(t_map_element* element)
//{
//	t_map_element* tmp;
//
//	tmp = element;
//	element = element->left;
//	tmp->left = element->right;
//	element->right = tmp;
//	element->weight = tmp->weight;
//	tmp->weight = tmp->left->weight + tmp->right->weight + 1;
//
//	return element;
//}
//
//static t_map_element* balance_binary_tree(t_map_element* element)
//{
//	float weight_balance;
//
//	if (element->left != NULL)
//	{
//		weight_balance = (float)element->left->weight / (float)element->weight;
//		if (weight_balance > 0.6)
//			return right_rotation(element);
//	}
//	else if (element->right != NULL)
//	{
//		weight_balance = (float)element->right->weight / (float)element->weight;
//		if (weight_balance > 0.6)
//			return left_rotation(element);
//	}
//
//	return element;
//
//	//if (weight_balance > 0.70771 && )
//	//if (element->left != NULL)
//	//{
//	//	weight_balance = (float)element->left->weight / (float)element->weight;
//	//	if (weight_balance > 0.70711 && element->left->left != NULL)
//	//	{
//	//		if ((float)element->left->left->weight / (float)element->left->weight <= 0.414213)
//	//			element->left = left_rotation(element->left);
//
//	//		element = right_rotation(element);
//	//	}
//	//	else if (weight_balance < 0.29289 && element->right != NULL && )
//	//}
//
//	//return element;
//
//	//float wbal = (float)root.left.weight / (float)root.weight;
//	//if (wbal > 0.70711 && root.left != nil) {
//
//	//	if ((float)root.left.left.weight / (float)root.left.weight > 0.414213)
//	//		root = rightRotate(root);
//	//	else {
//	//		root.left = leftRotate(root.left);
//	//		root = rightRotate(root);
//	//	}
//	//}
//	//else if (wbal < 0.29289 && root.right != nil) {
//	//	if ((float)(root.right.left.weight / (float)root.right.weight) < 0.585786)
//	//		root = leftRotate(root);
//	//	else {
//	//		root.right = rightRotate(root.right);
//	//		root = leftRotate(root);
//	//	}
//	//}
//	//return root;
//}
//
//
//static void display_binary_tree(t_map_element* element)
//{
//	if (element == NULL)
//		return;
//
//	display_binary_tree(element->left);
//	display_binary_tree(element->right);
//	printf("%d\n", element->key);
//}
//
//
//int init_map(t_map* map)
//{
//	if (pthread_mutex_init(&map->mutex, NULL) != 0)
//	{
//		fprintf(stderr, "Error in init_map: mutex init failed\n");
//		return -1;
//	}
//
//	map->root = NULL;
//
//	return 0;
//}
//
//void* get_map_element(t_map* map, int searched_key)
//{
//	t_map_element* tmp;
//
//	tmp = map->root;
//
//	while (tmp != NULL)
//	{
//		if (searched_key == tmp->key)
//			return tmp->data;
//		else if (searched_key < tmp->key)
//			tmp = tmp->left;
//		else
//			tmp = tmp->right;
//	}
//
//	return NULL;
//}
//
//void add_map_element(t_map* map, int key, void* data)
//{
//	t_map_element* tmp;
//
//	if (map->root == NULL)
//	{
//		map->root = get_memory(sizeof(t_map_element));
//		tmp = map->root;
//	}
//	else
//	{
//		map->root = balance_binary_tree(map->root);
//		tmp = map->root;
//		while (tmp != NULL)
//		{
//			tmp = balance_binary_tree(tmp);
//			++tmp->weight;
//
//			if (key < tmp->key)
//			{
//				if (tmp->left == NULL)
//				{
//					tmp->left = get_memory(sizeof(t_map_element));
//					tmp = tmp->left;
//					break;
//				}
//				else
//					tmp = tmp->left;
//			}
//			else
//			{
//				if (tmp->right == NULL)
//				{
//					tmp->right = get_memory(sizeof(t_map_element));
//					tmp = tmp->right;
//					break;
//				}
//				else
//					tmp = tmp->right;
//			}
//		}
//	}
//
//	tmp->key = key;
//	tmp->data = data;
//	tmp->weight = 1;
//	tmp->left = NULL;
//	tmp->right = NULL;
//}
//
//void display_map(t_map* map)
//{
//	display_binary_tree(map->root);
//}
//
////void *remove_map_element(t_map* map, int key)
////{
////	if (root == nil) {
////		System.out.println("Key not found");
////		return root;
////	}
////	if (key < root.element)
////		root.left = delete(key, root.left);
////	else if (key > root.element)
////		root.right = delete(key, root.right);
////
////	else if (root.left == nil)
////		root = root.right;  // root contains key and has one child - right
////	else if (root.right == nil)
////		root = root.left;   // root contains key and has one child - left
////
////	else if (root.left.weight > root.right.weight) {
////		root = rightRotate(root);
////		root.right = delete(key, root.right);
////	}
////	else {
////		root = leftRotate(root);
////		root.left = delete(key, root.left);
////	}
////
////	if (root != nil) {
////		root.weight = root.left.weight + root.right.weight + 1;
////		root = checkRotation(root);
////	}
////
////	return root;
////}