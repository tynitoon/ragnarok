#include <sys/mman.h>
#include <string.h>
#include <stdio.h>

#include <pthread.h>

#include "single_memory.h"
#include "list.h"

#define CHUNK_SIZE (1 << 27) //128 MegaBytes

typedef struct	s_block
{
	size_t		size;
	size_t		prev_size;
}				t_block;

static unsigned long	g_memory_index = 0;
static void*			g_memory_head = NULL;
static t_list			g_frees;
static pthread_mutex_t	g_main_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t	g_realloc_mutex = PTHREAD_MUTEX_INITIALIZER;

static void set_prev_size(t_block* block)
{
	size_t size = block->size;

	block = (t_block*)((unsigned long)block + sizeof(t_block) + size + sizeof(t_list_element));
	if ((unsigned long)block < (unsigned long)g_memory_head + CHUNK_SIZE)
		block->prev_size = size;
}

void* get_memory(size_t size)
{
	t_list_element*			new_element;
	t_list_element*			element;
	t_block*				new_block;
	t_block*				block;

	if (size == 0)
	{
		fprintf(stderr, "Error in get_memory: try to allocate 0 bytes\n");
		return NULL;
	}

	pthread_mutex_lock(&g_main_mutex);

	if (g_memory_head == NULL)
	{
		if (size > CHUNK_SIZE - sizeof(t_list_element) - sizeof(t_block))
		{
			pthread_mutex_unlock(&g_main_mutex);

			fprintf(stderr, "Error in get_memory: asked memory = %ld bytes, remaining memory = %ld bytes\n", size, CHUNK_SIZE - sizeof(t_list_element) - sizeof(t_block));
			return NULL;
		}

		g_memory_head = mmap(NULL, CHUNK_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);
		if (g_memory_head == MAP_FAILED)
		{
			g_memory_head = NULL;

			pthread_mutex_unlock(&g_main_mutex);

			fprintf(stderr, "Error in get_memory: mmap failed\n");
			return NULL;
		}

		memset(&g_frees, 0, sizeof(t_list));
		new_element = g_memory_head;
		new_element->next = g_memory_head - 1; //Impossible value, so we know it's not free
		new_block = (t_block*)new_element->data;
		new_block->size = size;
		new_block->prev_size = 0;

		g_memory_index = sizeof(t_list_element) + sizeof(t_block) + size;

		set_prev_size(new_block);

		pthread_mutex_unlock(&g_main_mutex);

		return new_element->data + sizeof(t_block);
	}

	//Search in free list
	element = g_frees.head;
	while (element != NULL)
	{
		block = (t_block*)element->data;
		if (block->size == size)
		{
			remove_list_element(&g_frees, element);
			element->next = g_memory_head - 1;

			pthread_mutex_unlock(&g_main_mutex);

			return element->data + sizeof(t_block);
		}
		else if (block->size > size + sizeof(t_list_element) + sizeof(t_block))
		{	
			new_element = (t_list_element*)((unsigned long)element + sizeof(t_list_element) + sizeof(t_block) + size);
			new_block = (t_block*)new_element->data;
			new_block->size = block->size - sizeof(t_list_element) - sizeof(t_block) - size;
			new_block->prev_size = size;

			set_prev_size(new_block);

			remove_list_element(&g_frees, element);
			add_list_element(&g_frees, new_element);

			element->next = g_memory_head - 1;
			block->size = size;

			pthread_mutex_unlock(&g_main_mutex);

			return element->data + sizeof(t_block);
		}
		element = element->next;
	}

	//There is no place in free list. We have to create a new t_list_element
	new_element = (t_list_element*)(g_memory_head + g_memory_index);

	if ((unsigned long)new_element + sizeof(t_block) + size > (unsigned long)g_memory_head + CHUNK_SIZE)
	{
		pthread_mutex_unlock(&g_main_mutex);

		fprintf(stderr, "Error in get_memory: memory is full\n");
		return NULL;
	}

	new_block = (t_block*)new_element->data;
	new_block->size = size;
	new_element->next = g_memory_head - 1;

	set_prev_size(new_block);

	g_memory_index += sizeof(t_list_element) + sizeof(t_block) + size;

	pthread_mutex_unlock(&g_main_mutex);

	return new_element->data + sizeof(t_block);
}

void* realloc_memory(void* ptr, size_t size)
{
	void*		new_ptr;
	t_block*	current_block;


	pthread_mutex_lock(&g_realloc_mutex);

	if (ptr <= g_memory_head || g_memory_head + CHUNK_SIZE < ptr)
	{
		pthread_mutex_unlock(&g_realloc_mutex);

		new_ptr = get_memory(size);

		return new_ptr;
	}

	current_block = (t_block*)(ptr - sizeof(t_block));

	if (size == current_block->size)
	{
		pthread_mutex_unlock(&g_realloc_mutex);

		return ptr;
	}

	new_ptr = get_memory(size);
	if (size > current_block->size)
		memcpy(new_ptr, ptr, current_block->size);
	else
		memcpy(new_ptr, ptr, size);

	free_memory(ptr);

	pthread_mutex_unlock(&g_realloc_mutex);

	return new_ptr;
}

void free_memory(void* ptr)
{
	t_list_element*		prev_element = NULL;
	t_list_element*		current_element;
	t_list_element*		next_element = NULL;
	t_block*			prev_block = NULL;
	t_block*			current_block;
	t_block*			next_block = NULL;

	pthread_mutex_lock(&g_main_mutex);

	if (ptr <= g_memory_head || g_memory_head + CHUNK_SIZE < ptr)
	{
		fprintf(stderr, "Error in free_memory: pointer is not allocated\n");
		return;
	}

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
		prev_block->size += current_block->size + next_block->size + 2 * sizeof(t_list_element) + 2 * sizeof(t_block);

		set_prev_size(prev_block);

		remove_list_element(&g_frees, prev_element);
		remove_list_element(&g_frees, next_element);
		add_list_element(&g_frees, prev_element);
	}
	else if (prev_element != NULL && prev_element->next != g_memory_head - 1)
	{
		prev_block->size += current_block->size + sizeof(t_list_element) + sizeof(t_block);

		set_prev_size(prev_block);
	}
	else if (next_element != NULL && next_element->next != g_memory_head - 1)
	{
		//Free block is before a free, so we took its place
		current_block->size += next_block->size + sizeof(t_list_element) + sizeof(t_block);
		current_element->prev = next_element->prev;
		current_element->next = next_element->next;

		set_prev_size(current_block);

		if (current_element->prev != NULL)
			current_element->prev->next = current_element;
		if (current_element->next != NULL)
			current_element->next->prev = current_element;

		if (g_frees.head == next_element)
			g_frees.head = current_element;
		if (g_frees.tail == next_element)
			g_frees.tail = current_element;
	}
	else
		add_list_element(&g_frees, current_element);

	pthread_mutex_unlock(&g_main_mutex);
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

	element = g_frees.head;
	printf("\nfree list :\n\n");
	while (element != NULL)
	{
		block = (t_block*)element->data;

		printf("element = %ld size = %ld free = %d element->prev = %ld\n", (unsigned long)element, block->size, (element->next != g_memory_head - 1), (unsigned long)element->prev);

		element = element->next;
	}

	printf("\nTotal allocated size = %ld\n", total_allocated_size);
}