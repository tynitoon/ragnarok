#include <sys/mman.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#include "single_memory.h"
#include "list.h"

#define CHUNK_SIZE (1 << 26) //64 MegaBytes

typedef struct	s_block
{
	size_t		size;
	size_t		prev_size;
}				t_block;

static unsigned long	g_memory_index = 0;
static void*			g_memory_head = NULL;
static t_list			g_frees = { .head = NULL, .tail = NULL };

void* get_memory(size_t size)
{
	static unsigned long	last_size = 0;
	t_list_element*			new_element;
	t_list_element*			element;
	t_block*				new_block;
	t_block*				block;
	t_block*				tmp_block;

	if (size == 0)
	{
		fprintf(stderr, "Error in get_memory: try to allocate 0 bytes\n");
		return NULL;
	}

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
			fprintf(stderr, "Error in get_memory: mmap failed\n");
			return NULL;
		}

		new_element = g_memory_head;
		new_element->next = g_memory_head - 1; //Impossible value, so we know it's not free
		new_block = (t_block*)new_element->buffer;
		new_block->size = size;
		new_block->prev_size = 0;

		g_memory_index = sizeof(t_list_element) + sizeof(t_block) + size;
		last_size = size;

		return new_element->buffer + sizeof(t_block);
	}

	//Search in free list
	element = g_frees.head;
	while (element != NULL)
	{
		block = (t_block*)element->buffer;
		if (block->size == size)
		{
			remove_list_element(&g_frees, element);
			element->next = g_memory_head - 1;

			return element->buffer + sizeof(t_block);
		}
		else if (block->size > size + sizeof(t_list_element) + sizeof(t_block))
		{	
			new_element = (t_list_element*)((unsigned long)element + sizeof(t_list_element) + sizeof(t_block) + size);
			new_block = (t_block*)new_element->buffer;
			new_block->size = block->size - sizeof(t_list_element) - sizeof(t_block) - size;
			new_block->prev_size = size;

			//printf("new element = %ld new size = %ld new prev size = %ld\n", (unsigned long)new_element, new_block->size, new_block->prev_size);

			tmp_block = (t_block*)((unsigned long)new_block + sizeof(t_block) + new_block->size + sizeof(t_list_element));
			if ((unsigned long)tmp_block < (unsigned long)g_memory_head + g_memory_index)
				tmp_block->prev_size = new_block->size;

			remove_list_element(&g_frees, element);
			add_list_element(&g_frees, new_element);

			element->next = g_memory_head - 1;
			block->size = size;

			return element->buffer + sizeof(t_block);
		}
		element = element->next;
	}

	//There is no place in free list. We have to create a new t_list_element
	new_element = (t_list_element*)(g_memory_head + g_memory_index);

	if ((unsigned long)new_element + sizeof(t_block) + size > (unsigned long)g_memory_head + CHUNK_SIZE)
	{
		fprintf(stderr, "Error in get_memory: memory is full\n");
		return NULL;
	}

	new_block = (t_block*)new_element->buffer;
	new_block->size = size;
	new_block->prev_size = last_size;
	new_element->next = g_memory_head - 1;

	g_memory_index += sizeof(t_list_element) + sizeof(t_block) + size;
	last_size = size;

	return new_element->buffer + sizeof(t_block);
}

void free_memory(void* ptr)
{
	t_list_element*		prev_element = NULL;
	t_list_element*		current_element;
	t_list_element*		next_element = NULL;
	t_block*			prev_block = NULL;
	t_block*			current_block;
	t_block*			next_block = NULL;
	t_block*			tmp_block;

	if (ptr == NULL || g_memory_head == NULL)
	{
		//fprintf(stderr, "Error in free_memory: try to free NULL pointer\n");
		return;
	}

	current_element = ptr - sizeof(t_list_element) - sizeof(t_block);
	current_block = (t_block*)current_element->buffer;

	if (current_block->prev_size != 0)
	{
		prev_element = (t_list_element*)((unsigned long)current_element - current_block->prev_size - sizeof(t_block) - sizeof(t_list_element));
		prev_block = (t_block*)prev_element->buffer;
	}

	//Check if the next element is outside the allocated memory
	if ((unsigned long)current_block + sizeof(t_block) + current_block->size < (unsigned long)g_memory_head + g_memory_index)
	{
		next_element = (t_list_element*)((unsigned long)current_block + sizeof(t_block) + current_block->size);
		next_block = (t_block*)next_element->buffer;
	}

	if (prev_element != NULL && next_element != NULL && prev_element->next != g_memory_head - 1 && next_element->next != g_memory_head - 1)
	{
		prev_block->size += current_block->size + next_block->size + 2 * sizeof(t_list_element) + 2 * sizeof(t_block);

		tmp_block = (t_block*)((unsigned long)next_block + sizeof(t_block) + next_block->size + sizeof(t_list_element));
		if ((unsigned long)tmp_block < (unsigned long)g_memory_head + g_memory_index)
			tmp_block->prev_size = prev_block->size;

		remove_list_element(&g_frees, prev_element);
		remove_list_element(&g_frees, next_element);
		add_list_element(&g_frees, prev_element);
	}
	else if (prev_element != NULL && prev_element->next != g_memory_head - 1)
	{
		prev_block->size += current_block->size + sizeof(t_list_element) + sizeof(t_block);
		if (next_block != NULL)
			next_block->prev_size = prev_block->size;
	}
	else if (next_element != NULL && next_element->next != g_memory_head - 1)
	{
		//Free block is before a free, so we took its place
		current_block->size += next_block->size + sizeof(t_list_element) + sizeof(t_block);
		current_element->prev = next_element->prev;
		current_element->next = next_element->next;

		tmp_block = (t_block*)((unsigned long)next_block + sizeof(t_block) + next_block->size + sizeof(t_list_element));
		if ((unsigned long)tmp_block < (unsigned long)g_memory_head + g_memory_index)
			tmp_block->prev_size = current_block->size;

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
}

void display_memory()
{
	t_list_element*		element;
	t_block*			block;

	element = g_memory_head;
	printf("Block list :\n\n");
	while (element != NULL)
	{
		block = (t_block*)element->buffer;
		if (block->size == 0)
			break;
		printf("element = %ld size = %ld bytes free = %d next block should be = %ld\n", (unsigned long)element, block->size, (element->next != g_memory_head - 1), (unsigned long)element + block->size + sizeof(t_list_element) + sizeof(t_block));
		element = (t_list_element*)((unsigned long)element + block->size + sizeof(t_list_element) + sizeof(t_block));
		if ((unsigned long)element >= (unsigned long)g_memory_head + CHUNK_SIZE)
			element = NULL;
	}

	element = g_frees.head;
	printf("\nfree list :\n\n");
	while (element != NULL)
	{
		block = (t_block*)element->buffer;
		printf("element = %ld size = %ld free = %d element->prev = %ld\n", (unsigned long)element, block->size, (element->next != g_memory_head - 1), (unsigned long)element->prev);
		element = element->next;
	}
}

void release_memory()
{
	if (g_memory_head == NULL)
		return;

	if (munmap(g_memory_head, CHUNK_SIZE) != 0)
		fprintf(stderr, "Error in release_memory: munmap failed\n");
}