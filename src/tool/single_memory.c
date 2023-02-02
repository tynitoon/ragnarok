#include <sys/mman.h>
#include <string.h>
#include <stdio.h>

#include <pthread.h>

#include "single_memory.h"

#define CHUNK_SIZE		(1 << 30) //1 Go
#define MAX_FREE_INDEX	20

typedef struct				s_block
{
	size_t					size;
	size_t					prev_size;
	struct s_block*			next_free;
	struct s_block*			prev_free;
}							t_block;

static unsigned long	g_memory_index = 0;
static void*			g_memory_head = NULL;
static t_block*			g_frees[MAX_FREE_INDEX];
static int				g_table_index[MAX_FREE_INDEX];
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

	return tab64[((size_t)((size - (size >> 1)) * 0x07EDD5E59A4E28C2)) >> 58] >> 1;
}

static size_t align_size(size_t size)
{
	return (size + (sizeof(void*) - 1)) & -sizeof(void*);
}

static void add_list_element(t_block* to_add, int index)
{
	int	i;

	to_add->prev_free = NULL;
	to_add->next_free = g_frees[index];

	if (g_frees[index] != NULL)
		g_frees[index]->prev_free = to_add;

	g_frees[index] = to_add;

	for (i = 0; i <= index; ++i)
	{
		if (g_table_index[i] == -1 || index < g_table_index[i])
			g_table_index[i] = index;
	}
}

static void remove_list_element(t_block* to_remove, int index)
{
	int	i;
	int	max_index;

	if (to_remove == NULL)
		return;

	if (g_frees[index] == to_remove)
	{
		g_frees[index] = to_remove->next_free;
		if (g_frees[index] == NULL)
		{
			max_index = g_table_index[index + 1];
			for (i = 0; i <= index; ++i)
			{
				if (g_table_index[i] == index)
					g_table_index[i] = max_index;
			}
		}
	}
	else
	{
		to_remove->prev_free->next_free = to_remove->next_free;

		if (to_remove->next_free != NULL)
			to_remove->next_free->prev_free = to_remove->prev_free;
	}

	to_remove->next_free = g_memory_head - 1;
}

static void set_prev_size(t_block* block)
{
	size_t size = block->size;

	block = (t_block*)((unsigned long)block + sizeof(t_block) + size);
	if ((unsigned long)block < (unsigned long)g_memory_head + CHUNK_SIZE)
		block->prev_size = size;
}

static void split_memory(t_block* block, size_t size)
{
	t_block* new_block;

	if (block->next_free != g_memory_head - 1)
		remove_list_element(block, compute_index(block->size));

	new_block = (t_block*)((unsigned long)block + sizeof(t_block) + size);
	new_block->size = block->size - sizeof(t_block) - size;
	new_block->prev_size = size;

	set_prev_size(new_block);

	add_list_element(new_block, compute_index(new_block->size));
	
	block->size = size;
}

static void* get_memory_unsafe(size_t size)
{
	t_block*		new_block;
	t_block*		block;
	int				index;

	if (size == 0)
		return NULL;

	if (g_memory_head == NULL)
	{
		if (size > CHUNK_SIZE - sizeof(t_block))
		{
			fprintf(stderr, "Error in get_memory: asked memory = %ld bytes, remaining memory = %ld bytes\n", size, CHUNK_SIZE - sizeof(t_block));
			return NULL;
		}

		g_memory_head = mmap(NULL, CHUNK_SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);

		if (g_memory_head == MAP_FAILED)
		{
			g_memory_head = NULL;

			fprintf(stderr, "Error in get_memory: mmap failed\n");
			return NULL;
		}

		memset(&g_frees, 0, sizeof(t_block*) * MAX_FREE_INDEX);
		memset(&g_table_index, -1, sizeof(int) * MAX_FREE_INDEX);
		new_block = g_memory_head;
		new_block->next_free = g_memory_head - 1; //Impossible value, so we know it's not free
		new_block->size = size;
		new_block->prev_size = 0;

		g_memory_index = sizeof(t_block) + size;

		set_prev_size(new_block);

		return (void*)((unsigned long)new_block + sizeof(t_block));
	}

	//Search in free list
	index = g_table_index[compute_index(size) + 1];

	if (index != -1)
	{
		block = g_frees[index];
		if (block != NULL)
		{
			if (block->size > sizeof(t_block) + size)
			{
				split_memory(block, size);

				return (void*)((unsigned long)block + sizeof(t_block));
			}
			else
			{
				remove_list_element(block, index);

				return (void*)((unsigned long)block + sizeof(t_block));
			}
		}
	}

	//There is no place in free list. We have to create a new t_list_element
	new_block = (t_block*)((unsigned long)g_memory_head + g_memory_index);

	if ((unsigned long)new_block + sizeof(t_block) + size > (unsigned long)g_memory_head + CHUNK_SIZE)
	{
		fprintf(stderr, "Error in get_memory: memory is full\n");
		return NULL;
	}

	new_block->size = size;
	new_block->next_free = g_memory_head - 1;

	set_prev_size(new_block);

	g_memory_index += sizeof(t_block) + size;

	return (void*)((unsigned long)new_block + sizeof(t_block));
}

static void free_memory_unsafe(void* ptr)
{
	t_block*		prev_block = NULL;
	t_block*		current_block;
	t_block*		next_block = NULL;
	int				oldIndex;
	int				newIndex;

	if (g_memory_head == NULL || ptr < g_memory_head || g_memory_head + CHUNK_SIZE < ptr)
		return;

	current_block = (t_block*)((unsigned long)ptr - sizeof(t_block));

	if (current_block->prev_size != 0)
		prev_block = (t_block*)((unsigned long)current_block - sizeof(t_block) - current_block->prev_size);

	//Check if the next element is outside the allocated memory
	if ((unsigned long)current_block + sizeof(t_block) + current_block->size < (unsigned long)g_memory_head + g_memory_index)
		next_block = (t_block*)((unsigned long)current_block + sizeof(t_block) + current_block->size);

	if (prev_block != NULL && next_block != NULL && prev_block->next_free != g_memory_head - 1 && next_block->next_free != g_memory_head - 1)
	{
		oldIndex = compute_index(prev_block->size);
		prev_block->size += (sizeof(t_block) << 1) + current_block->size + next_block->size;
		newIndex = compute_index(prev_block->size);

		set_prev_size(prev_block);

		remove_list_element(next_block, compute_index(next_block->size));
		if (oldIndex != newIndex)
		{
			remove_list_element(prev_block, oldIndex);
			add_list_element(prev_block, newIndex);
		}
	}
	else if (prev_block != NULL && prev_block->next_free != g_memory_head - 1)
	{
		oldIndex = compute_index(prev_block->size);
		prev_block->size += sizeof(t_block) + current_block->size;
		newIndex = compute_index(prev_block->size);

		set_prev_size(prev_block);

		if (oldIndex != newIndex)
		{
			remove_list_element(prev_block, oldIndex);
			add_list_element(prev_block, newIndex);
		}
	}
	else if (next_block != NULL && next_block->next_free != g_memory_head - 1)
	{
		remove_list_element(next_block, compute_index(next_block->size));
		current_block->size += sizeof(t_block) + next_block->size;
		add_list_element(current_block, compute_index(current_block->size));
	}
	else
		add_list_element(current_block, compute_index(current_block->size));
}

void* get_memory(size_t size)
{
	void*	ptr;

	pthread_mutex_lock(&g_main_mutex);

	size = align_size(size);
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

	size = align_size(size);
	current_block = (t_block*)((unsigned long)ptr - sizeof(t_block));

	if (current_block->size > sizeof(t_block) + size)
	{
		split_memory(current_block, size);

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
	t_block*		block;
	size_t			total_allocated_size = 0;
	int				i;

	block = (t_block*)g_memory_head;
	printf("Block list :\n\n");
	while (block != NULL)
	{
		if (block->size == 0)
			break;

		printf("element = %ld size = %ld bytes free = %d next block should be = %ld\n", (unsigned long)block, block->size, (block->next_free != g_memory_head - 1), (unsigned long)block + sizeof(t_block) + block->size);

		if (block->next_free == g_memory_head - 1)
			total_allocated_size += sizeof(t_block) + block->size;

		block = (t_block*)((unsigned long)block + sizeof(t_block) + block->size);
		if ((unsigned long)block >= (unsigned long)g_memory_head + CHUNK_SIZE)
			block = NULL;
	}

	printf("\nfree list :\n\n");
	for (i = 0; i < MAX_FREE_INDEX; ++i)
	{
		block = g_frees[i];
		while (block != NULL)
		{
			printf("element = %ld size = %ld free = %d element->prev = %ld\n", (unsigned long)block, block->size, (block->next_free != g_memory_head - 1), (unsigned long)block->prev_free);

			block = block->next_free;
		}
	}

	printf("\nTotal allocated size = %ld\n", total_allocated_size);
}