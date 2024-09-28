#include <assert.h>
#include <stdio.h>

#include "single_memory.h"
#include "list.h"

#define DATA_NB 100

int main()
{
	int data[DATA_NB];

	for (int i = 0; i < DATA_NB; ++i)
		data[i] = i;

	t_list list;
	list_init(&list);

	printf("Test list_add_front:\n");
	list_add_front(&list, &data[0]);
	assert(list_find(&list, &data[0]) == 0);
	list_add_front(&list, &data[1]);
	assert(list_find(&list, &data[0]) == 1);
	assert(list_find(&list, &data[1]) == 0);
	printf("OK!\n\n");

	printf("Test list_add_back:\n");
	list_add_back(&list, &data[2]);
	assert(list_find(&list, &data[2]) == 2);
	list_add_back(&list, &data[3]);
	assert(list_find(&list, &data[2]) == 2);
	assert(list_find(&list, &data[3]) == 3);
	printf("OK!\n\n");


	printf("Test list_add_at:\n");
	list_add_at(&list, &data[4], 0);
	assert(list_find(&list, &data[4]) == 0);
	list_add_at(&list, &data[5], 5);
	assert(list_find(&list, &data[5]) == 5);
	list_add_at(&list, &data[6], 1);
	assert(list_find(&list, &data[6]) == 1);
	list_add_at(&list, &data[7], 6);
	assert(list_find(&list, &data[7]) == 6);
	printf("OK!\n\n");

	printf("Test list_remove_front:\n");
	assert(*(int*)list_remove_front(&list) == data[4]);
	assert(*(int*)list_remove_back(&list) == data[5]);
	assert(*(int*)list_remove_at(&list, 0) == data[6]);
	assert(*(int*)list_remove_at(&list, 4) == data[7]);
	assert(*(int*)list_remove_at(&list, 1) == data[0]);
	printf("OK!\n\n");

	printf("Test list_display:\n");
	list_display(&list);
	printf("OK!\n\n");

	printf("Test list_clear:\n");
	list_clear(&list);
	assert(list.head == NULL && list.tail == NULL && list.nb_elements == 0);
	printf("OK!\n\n");

	return 0;
}