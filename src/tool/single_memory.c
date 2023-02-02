#include <sys/mman.h>
#include <string.h>
#include <stdio.h>

#include <pthread.h>

#include "single_memory.h"

#define CHUNK_SIZE		(1 << 27) //128 MegaBytes
#define MAX_FREE_INDEX	15

typedef struct				s_block
{
	size_t					size;
	size_t					prev_size;
}							t_block;

//We have to reimplement the list to remove the mutex
typedef struct              s_list_element
{
	struct s_list_element*	prev;
	struct s_list_element*	next;
	char                    data[];
}                           t_list_element;

typedef struct              s_list
{
	t_list_element*			head;
	t_list_element*			tail;
}                           t_list;

static unsigned long	g_memory_index = 0;
static void*			g_memory_head = NULL;
static t_list			g_frees[MAX_FREE_INDEX];
static pthread_mutex_t	g_main_mutex = PTHREAD_MUTEX_INITIALIZER;

static int	compute_index(size_t size)
{
	static const int tab64[64] =
	{
		63,  0, 58,  1, 59, 47, 53,  2,
		60, 39, 48, 27, 54, 33, 42,  3,
		61, 51, 37, 40, 49, 18, 28, 20,
		55, 30, 34, 11, 43, 14, 22,  4,
		62, 57, 46, 52, 38, 26, 32, 41,
		50, 36, 17, 19, 29, 10, 13, 21,
		56, 45, 25, 31, 35, 16,  9, 12,
		44, 24, 15,  8, 23,  7,  6,  5
	};

	size |= size >> 1;
	size |= size >> 2;
	size |= size >> 4;
	size |= size >> 8;
	size |= size >> 16;
	size |= size >> 32;

	return tab64[((size_t)((size - (size >> 1)) * 0x07EDD5E59A4E28C2)) >> 58] / 2;
}

static void add_list_element(t_list_element* to_add, int index)
{
	to_add->next = NULL;
	to_add->prev = g_frees[index].tail;

	if (g_frees[index].tail == NULL)
	{
		g_frees[index].tail = to_add;
		g_frees[index].head = to_add;
	}
	else
	{
		g_frees[index].tail->next = to_add;
		g_frees[index].tail = to_add;
	}
}

static void remove_list_element(t_list_element* to_remove, int index)
{
	if (to_remove == NULL)
		return;

	if (g_frees[index].head == to_remove)
	{
		g_frees[index].head = to_remove->next;
		if (g_frees[index].head != NULL)
			g_frees[index].head->prev = NULL;
		else
			g_frees[index].tail = NULL;
	}
	else if (g_frees[index].tail == to_remove)
	{
		g_frees[index].tail = to_remove->prev;
		if (g_frees[index].tail != NULL)
			g_frees[index].tail->next = NULL;
		else
			g_frees[index].head = NULL;
	}
	else
	{
		to_remove->prev->next = to_remove->next;
		to_remove->next->prev = to_remove->prev;
	}

	to_remove->next = g_memory_head - 1;
}

static void set_prev_size(t_block* block)
{
	size_t size = block->size;

	block = (t_block*)((unsigned long)block + sizeof(t_block) + size + sizeof(t_list_element));
	if ((unsigned long)block < (unsigned long)g_memory_head + CHUNK_SIZE)
		block->prev_size = size;
}

static void split_memory(t_list_element* element, t_block* block, size_t size)
{
	t_list_element* new_element;
	t_block* new_block;

	if (element->next != g_memory_head - 1)
		remove_list_element(element, compute_index(block->size));

	new_element = (t_list_element*)((unsigned long)element + sizeof(t_list_element) + sizeof(t_block) + size);
	new_block = (t_block*)new_element->data;
	new_block->size = block->size - sizeof(t_list_element) - sizeof(t_block) - size;
	new_block->prev_size = size;

	set_prev_size(new_block);

	add_list_element(new_element, compute_index(new_block->size));
	
	block->size = size;
}

static void* get_memory_unsafe(size_t size)
{
	t_list_element* new_element;
	t_list_element* element;
	t_block*		new_block;
	t_block*		block;
	int				index;

	if (size == 0)
		return NULL;

	if (g_memory_head == NULL)
	{
		if (size > CHUNK_SIZE - sizeof(t_list_element) - sizeof(t_block))
		{
			fprintf(stderr, "Error in get_memory: asked memory = %ld bytes, remaining memory = %ld bytes\n", size, CHUNK_SIZE - sizeof(t_list_element) - sizeof(t_block));
			return NULL;
		}

		g_memory_head = mmap(NULL, CHUNK_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);
		if (g_memory_head == MAP_FAILED)
		{
			g_memory_head = NULL;

			fprintf(stderr, "Error in get_memory: mmap failed\n");
			return NULL;
		}

		memset(&g_frees, 0, sizeof(g_frees));
		new_element = g_memory_head;
		new_element->next = g_memory_head - 1; //Impossible value, so we know it's not free
		new_block = (t_block*)new_element->data;
		new_block->size = size;
		new_block->prev_size = 0;

		g_memory_index = sizeof(t_list_element) + sizeof(t_block) + size;

		set_prev_size(new_block);

		return new_element->data + sizeof(t_block);
	}

	//Search in free list
	index = compute_index(size) + 1;
	if (index >= MAX_FREE_INDEX)
		index = MAX_FREE_INDEX - 1;

	while (index < MAX_FREE_INDEX)
	{
		element = g_frees[index].head;
		if (element != NULL)
		{
			block = (t_block*)element->data;
			if (block->size > size + sizeof(t_list_element) + sizeof(t_block))
			{
				split_memory(element, block, size);

				return element->data + sizeof(t_block);
			}
			else
			{
				remove_list_element(element, index);

				return element->data + sizeof(t_block);
			}
		}
		++index;
	}

	//There is no place in free list. We have to create a new t_list_element
	new_element = (t_list_element*)(g_memory_head + g_memory_index);

	if ((unsigned long)new_element + sizeof(t_block) + size > (unsigned long)g_memory_head + CHUNK_SIZE)
	{
		fprintf(stderr, "Error in get_memory: memory is full\n");
		return NULL;
	}

	new_block = (t_block*)new_element->data;
	new_block->size = size;
	new_element->next = g_memory_head - 1;

	set_prev_size(new_block);

	g_memory_index += sizeof(t_list_element) + sizeof(t_block) + size;

	return new_element->data + sizeof(t_block);
}

static void free_memory_unsafe(void* ptr)
{
	t_list_element* prev_element = NULL;
	t_list_element* current_element;
	t_list_element* next_element = NULL;
	t_block*		prev_block = NULL;
	t_block*		current_block;
	t_block*		next_block = NULL;
	int				oldIndex;
	int				newIndex;

	if (g_memory_head == NULL || ptr < g_memory_head || g_memory_head + CHUNK_SIZE < ptr)
		return;

	current_element = ptr - sizeof(t_list_element) - sizeof(t_block);
	current_block = (t_block*)current_element->data;

	if (current_block->prev_size != 0)
	{
		prev_element = (t_list_element*)((unsigned long)current_element - current_block->prev_size - sizeof(t_block) - sizeof(t_list_element));
		prev_block = (t_block*)prev_element->data;
	}

	//Check if the next element is outside the allocated memory
	if ((unsigned long)current_block + sizeof(t_block) + current_block->size < (unsigned long)g_memory_head + g_memory_index)
	{
		next_element = (t_list_element*)((unsigned long)current_block + sizeof(t_block) + current_block->size);
		next_block = (t_block*)next_element->data;
	}

	if (prev_element != NULL && next_element != NULL && prev_element->next != g_memory_head - 1 && next_element->next != g_memory_head - 1)
	{
		oldIndex = compute_index(prev_block->size);
		prev_block->size += current_block->size + next_block->size + 2 * sizeof(t_list_element) + 2 * sizeof(t_block);
		newIndex = compute_index(prev_block->size);

		set_prev_size(prev_block);

		remove_list_element(next_element, compute_index(next_block->size));
		if (oldIndex != newIndex)
		{
			remove_list_element(prev_element, oldIndex);
			add_list_element(prev_element, newIndex);
		}
	}
	else if (prev_element != NULL && prev_element->next != g_memory_head - 1)
	{
		oldIndex = compute_index(prev_block->size);
		prev_block->size += current_block->size + sizeof(t_list_element) + sizeof(t_block);
		newIndex = compute_index(prev_block->size);

		set_prev_size(prev_block);

		if (oldIndex != newIndex)
		{
			remove_list_element(prev_element, oldIndex);
			add_list_element(prev_element, newIndex);
		}
	}
	else if (next_element != NULL && next_element->next != g_memory_head - 1)
	{
		remove_list_element(next_element, compute_index(next_block->size));
		current_block->size += next_block->size + sizeof(t_list_element) + sizeof(t_block);
		add_list_element(current_element, compute_index(current_block->size));
	}
	else
		add_list_element(current_element, compute_index(current_block->size));
}

void* get_memory(size_t size)
{
	void*	ptr;

	pthread_mutex_lock(&g_main_mutex);

	ptr = get_memory_unsafe(size);

	pthread_mutex_unlock(&g_main_mutex);

	return ptr;
}

void free_memory(void* ptr)
{
	pthread_mutex_lock(&g_main_mutex);

	free_memory_unsafe(ptr);

	pthread_mutex_unlock(&g_main_mutex);
}

void* realloc_memory(void* ptr, size_t size)
{
	void*		new_ptr;
	t_block*	current_block;

	if (ptr < g_memory_head || g_memory_head + CHUNK_SIZE < ptr)
	{
		new_ptr = get_memory_unsafe(size);

		pthread_mutex_unlock(&g_main_mutex);

		return new_ptr;
	}

	current_block = (t_block*)((unsigned long)ptr - sizeof(t_block));

	if (current_block->size > size + sizeof(t_list_element) + sizeof(t_block))
	{
		split_memory((t_list_element*)((unsigned long)current_block - sizeof(t_list_element)), current_block, size);

		pthread_mutex_unlock(&g_main_mutex);

		return ptr;
	}
	else if (current_block->size >= size)
	{
		pthread_mutex_unlock(&g_main_mutex);

		return ptr;
	}

	new_ptr = get_memory_unsafe(size);

	if (size > current_block->size)
		memcpy(new_ptr, ptr, current_block->size);
	else
		memcpy(new_ptr, ptr, size);
	
	free_memory_unsafe(ptr);

	pthread_mutex_unlock(&g_main_mutex);

	return new_ptr;
}

void release_memory()
{
	if (g_memory_head == NULL)
		return;

	if (munmap(g_memory_head, CHUNK_SIZE) != 0)
		fprintf(stderr, "Error in release_memory: munmap failed\n");
}

void display_memory()
{
	t_list_element* element;
	t_block*		block;
	size_t			total_allocated_size = 0;
	int				i;

	element = g_memory_head;
	printf("Block list :\n\n");
	while (element != NULL)
	{
		block = (t_block*)element->data;
		if (block->size == 0)
			break;

		printf("element = %ld size = %ld bytes free = %d next block should be = %ld\n", (unsigned long)element, block->size, (element->next != g_memory_head - 1), (unsigned long)element + block->size + sizeof(t_list_element) + sizeof(t_block));

		if (element->next == g_memory_head - 1)
			total_allocated_size += (sizeof(t_list_element) + sizeof(t_block) + block->size);

		element = (t_list_element*)((unsigned long)element + block->size + sizeof(t_list_element) + sizeof(t_block));
		if ((unsigned long)element >= (unsigned long)g_memory_head + CHUNK_SIZE)
			element = NULL;
	}

	printf("\nfree list :\n\n");
	for (i = 0; i < MAX_FREE_INDEX; ++i)
	{
		element = g_frees[i].head;
		while (element != NULL)
		{
			block = (t_block*)element->data;

			printf("element = %ld size = %ld free = %d element->prev = %ld\n", (unsigned long)element, block->size, (element->next != g_memory_head - 1), (unsigned long)element->prev);

			element = element->next;
		}
	}

	printf("\nTotal allocated size = %ld\n", total_allocated_size);
}