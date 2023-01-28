#include <stdio.h>
#include <string.h>

#include "map.h"
#include "single_memory.h"
#include "hash.h"

#define CHUNK_SIZE (1 << 6) // 64 t_list

static void refresh_map(t_map* map)
{
	unsigned int	hash;
	t_list_element* list_element;
	t_map_element*	map_element;
	t_list			tmp_list;
	size_t			i;

	memset(&tmp_list, 0, sizeof(t_list));

	for (i = 0; i < map->size - CHUNK_SIZE; ++i)
	{
		list_element = map->datas[i].head;
		while (list_element != NULL)
		{
			add_list_element(&tmp_list, list_element);
			list_element = list_element->next;
		}
		map->datas[i].head = NULL;
		map->datas[i].tail = NULL;
	}

	list_element = tmp_list.head;
	while (list_element != NULL)
	{
		map_element = (t_map_element*)list_element->data;
		hash = generate_hash(map_element->key, map->key_size);
		add_list_element(&map->datas[hash % map->size], list_element);

		list_element = list_element->next;
	}
}

static void grow_map(t_map* map)
{
	map->datas = realloc_memory(map->datas, map->size * sizeof(t_list) + CHUNK_SIZE * sizeof(t_list));
	memset(&map->datas[map->size], 0, CHUNK_SIZE * sizeof(t_list));
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
	t_list_element* list_element;
	t_map_element*	map_element;

	pthread_mutex_lock(&map->mutex);

	if (map->size == 0)
		grow_map(map);

	hash = generate_hash(key, map->key_size);

	list_element = map->datas[hash % map->size].head;
	while (list_element != NULL)
	{
		map_element = (t_map_element*)list_element->data;
		if (map_element->key == key)
		{
			pthread_mutex_unlock(&map->mutex);

			return;
		}

		list_element = list_element->next;
	}

	list_element = get_memory(sizeof(t_list_element) + sizeof(t_map_element));
	map_element = (t_map_element*)list_element->data;
	map_element->key = key;
	map_element->data = data;
	add_list_element(&map->datas[hash % map->size], list_element);

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
	t_list_element* list_element;
	t_map_element*	map_element;

	pthread_mutex_lock(&map->mutex);

	if (map->size == 0)
	{
		pthread_mutex_unlock(&map->mutex);

		return NULL;
	}

	hash = generate_hash(key, map->key_size);

	list_element = map->datas[hash % map->size].head;
	while (list_element != NULL)
	{
		map_element = (t_map_element*)list_element->data;
		if (map_element->key == key)
		{
			pthread_mutex_unlock(&map->mutex);

			return map_element->data;
		}

		list_element = list_element->next;
	}

	pthread_mutex_unlock(&map->mutex);

	return NULL;
}

void* remove_map_element(t_map* map, void* key)
{
	unsigned int	hash;
	t_list_element* list_element;
	t_map_element* map_element;

	pthread_mutex_lock(&map->mutex);

	if (map->size == 0)
	{
		pthread_mutex_unlock(&map->mutex);

		return NULL;
	}

	hash = generate_hash(key, map->key_size);

	list_element = map->datas[hash % map->size].head;
	while (list_element != NULL)
	{
		map_element = (t_map_element*)list_element->data;
		if (map_element->key == key)
		{
			remove_list_element(&map->datas[hash % map->size], list_element);

			pthread_mutex_unlock(&map->mutex);

			return map_element->data;
		}

		list_element = list_element->next;
	}

	pthread_mutex_unlock(&map->mutex);

	return NULL;
}

void display_map(t_map* map)
{
	t_list_element* list_element;
	t_map_element*	map_element;
	size_t			i;

	for (i = 0; i < map->size; ++i)
	{
		list_element = map->datas[i].head;
		while (list_element != NULL)
		{
			map_element = (t_map_element*)list_element->data;
			printf("index = %ld, key = %p, key size = %ld, data = %p, hash = %u\n", i, map_element->key, map->key_size, map_element->data, generate_hash(map_element->key, map->key_size));

			list_element = list_element->next;
		}
	}
	printf("size map = %ld\n", map->size);
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