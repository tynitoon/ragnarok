#include <sys/mman.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <pthread.h>

#include "single_memory.h"

#define PAGE_NUMBER				18											//In this system a page is 4096 bytes, so 4096 * 2^18 = 1Go, mmap need multiple of a page size
#define MAX_FREE_INDEX			40											//Each index is a power of 2. So max size can be 2^40 (more than 1 000 000 000 000)
#define NOT_INITIALIZED_VALUE	100											//Used to init g_table_index so we just have to compare index with <

typedef enum				s_filter
{
	PREV_BLOCK_IS_FREE	= (1 << 0),
	NEXT_BLOCK_IS_FREE	= (1 << 1),
	BOTH_BLOCK_ARE_FREE = PREV_BLOCK_IS_FREE | NEXT_BLOCK_IS_FREE
}							t_filter;

typedef struct				s_block
{
	uint64_t				size;											//Navigate to the next block
	uint64_t				prev_size;										//Navigate to the previous block
	struct s_block*			next_free;										//Double linked list on free blocks. Equal g_impossible_address if not free or something else if it is
	struct s_block*			prev_free;										//Double linked list on free blocks
}							t_block;

static uint64_t				g_memory_size;									//Size of the mmap allocation
static uint64_t				g_memory_offset = 0;							//Max address offset, if it reachs g_memory_size, the memory can be full if no free block match
static void*				g_memory_head = NULL;							//Address of the head memory (given at the first call by mmap)
static t_block*				g_frees[MAX_FREE_INDEX];						//Free blocks are ordered in it, depending of the compute_index(block->size) 
static int					g_table_index[MAX_FREE_INDEX];					//A index in free blocks can be empty. We use this table to reduce iterations
static pthread_mutex_t		g_main_mutex = PTHREAD_MUTEX_INITIALIZER;		//For Thread safe allocation

static void* g_impossible_address = (void*)sizeof(void*);					//Impossible address, it used to detect a free element
																			//!!!WARNING!!! This variable can change randomly in gcc -O3 but it seems ok if MAX_FREE_INDEX is equal to 40 !!!WARNING!!!

//It is a quick Log2, it uses the Debruijn algorithm
static int	compute_index(uint64_t size)
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


	return tab64[((uint64_t)((size - (size >> 1)) * 0x07EDD5E59A4E28C2)) >> 58];
}

//Align value to the current architecture to gain performances, negative sizeof is used as mask, example with size == 5 : (5 + 8 - 1) & -8 => 0000 1100 & 1111 1000 => 0000 1000 == 8
//A negative number is full of 1, this integer 0b11111111111111111111111111111000 is equal to -8 and 0b11111111111111111111111111111111 is equal to -1
static uint64_t align_size(uint64_t size)
{
	return (size + sizeof(void*) - 1) & -sizeof(void*);
}

static void add_list_element(t_block* to_add, int index)
{
	//Push front
	to_add->prev_free = NULL;
	to_add->next_free = g_frees[index];

	if (g_frees[index] != NULL)
		g_frees[index]->prev_free = to_add;

	g_frees[index] = to_add;

	//Update g_table_index
	for (int i = 0; i <= index; ++i)
	{
		if (index < g_table_index[i])
			g_table_index[i] = index;
	}
}

static void remove_list_element(t_block* to_remove, int index)
{
	int	max_index;

	if (to_remove == NULL)
		return;

	if (g_frees[index] == to_remove)
	{
		g_frees[index] = to_remove->next_free;

		//Update g_table_index
		if (g_frees[index] == NULL)
		{
			max_index = g_table_index[index + 1];
			for (int i = 0; i <= index; ++i)
			{
				if (g_table_index[i] == index)
					g_table_index[i] = max_index;
			}
		}
	}
	else
	{
		//The variable prev_free is only used to remove a block from list without iteration
		to_remove->prev_free->next_free = to_remove->next_free;

		if (to_remove->next_free != NULL)
			to_remove->next_free->prev_free = to_remove->prev_free;
	}

	to_remove->next_free = (t_block*)g_impossible_address;
}

//We avoid problems by setting the prev_size variable before creating the a block
static void set_prev_size(t_block* block)
{
	uint64_t size = block->size;

	block = (t_block*)((uint64_t)block + sizeof(t_block) + size);
	if ((uint64_t)block < (uint64_t)g_memory_head + g_memory_size)
		block->prev_size = size;
}

//With a split, the get_memory is a kind of best fit
static void split_memory(t_block* block, uint64_t size)
{
	t_block* new_block;

	if (block->next_free != g_impossible_address)
		remove_list_element(block, compute_index(block->size));

	new_block = (t_block*)((uint64_t)block + sizeof(t_block) + size);
	new_block->size = block->size - sizeof(t_block) - size;
	new_block->prev_size = size;

	set_prev_size(new_block);

	add_list_element(new_block, compute_index(new_block->size));
	
	block->size = size;
}

//Get memory function without mutex
static void* get_memory_unsafe(uint64_t size)
{
	t_block*		new_block;
	t_block*		block;
	int				index;

	if (size == 0)
		return NULL;

	//Init memory if it is the first call
	if (g_memory_head == NULL)
	{
		g_memory_size = (uint64_t)sysconf(_SC_PAGE_SIZE) << PAGE_NUMBER;
		g_memory_head = mmap(NULL, g_memory_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);

		if (g_memory_head == MAP_FAILED)
		{
			g_memory_head = NULL;

			fprintf(stderr, "Error in get_memory: mmap failed\n");
			return NULL;
		}

		g_impossible_address = (void*)((uint64_t)g_memory_head - sizeof(void*));

		memset(&g_frees, 0, sizeof(t_block*) * MAX_FREE_INDEX);
		for (int i = 0; i < MAX_FREE_INDEX; ++i)
			g_table_index[i] = NOT_INITIALIZED_VALUE;

		new_block = (t_block*)g_memory_head;
		new_block->next_free = (t_block*)g_impossible_address;
		new_block->size = size;
		new_block->prev_size = 0;

		g_memory_offset = sizeof(t_block) + size;

		set_prev_size(new_block);

		//return the point on the data, not on the block
		return (void*)((uint64_t)new_block + sizeof(t_block));
	}

	//Avoid impossible values
	if (size > g_memory_size - sizeof(t_block))
		return NULL;

	//Search in free list
	index = g_table_index[compute_index(size) + 1];

	if (index != NOT_INITIALIZED_VALUE)
	{
		block = g_frees[index];
		if (block != NULL)
		{
			if (block->size > sizeof(t_block) + size)
			{
				split_memory(block, size);

				return (void*)((uint64_t)block + sizeof(t_block));
			}
			else
			{
				remove_list_element(block, index);

				return (void*)((uint64_t)block + sizeof(t_block));
			}
		}
	}

	//There is no place in free list. We have to create a new block using the offset
	new_block = (t_block*)((uint64_t)g_memory_head + g_memory_offset);

	if ((uint64_t)new_block + sizeof(t_block) + size > (uint64_t)g_memory_head + g_memory_size)
	{
		fprintf(stderr, "Error in get_memory: memory is full or asked size is too big\n");
		return NULL;
	}

	new_block->size = size;
	new_block->next_free = (t_block*)g_impossible_address;

	set_prev_size(new_block);

	//We update the offset
	g_memory_offset += sizeof(t_block) + size;

	return (void*)((uint64_t)new_block + sizeof(t_block));
}

static void free_memory_unsafe(void* ptr)
{
	t_block*		prev_block = NULL;
	t_block*		current_block;
	t_block*		next_block = NULL;
	int				oldIndex;
	int				newIndex;
	int				filter = 0;

	current_block = (t_block*)((uint64_t)ptr - sizeof(t_block));

	if (g_memory_head == NULL || ptr < g_memory_head || (uint64_t)g_memory_head + g_memory_size < (uint64_t)ptr)
		return;

	//Check if previous block exists and get it if it does
	if (current_block->prev_size != 0)
	{
		prev_block = (t_block*)((uint64_t)current_block - sizeof(t_block) - current_block->prev_size);
		if (prev_block->next_free != g_impossible_address)
			filter |= PREV_BLOCK_IS_FREE;
	}

	//Check if the next block is inside the allocated memory and get it if it is
	if ((uint64_t)current_block + sizeof(t_block) + current_block->size < (uint64_t)g_memory_head + g_memory_offset)
	{
		next_block = (t_block*)((uint64_t)current_block + sizeof(t_block) + current_block->size);
		if (next_block->next_free != g_impossible_address)
			filter |= NEXT_BLOCK_IS_FREE;
	}

	//Try to merge blocks
	switch (filter)
	{
		case BOTH_BLOCK_ARE_FREE:
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
			break;
		case PREV_BLOCK_IS_FREE:
			oldIndex = compute_index(prev_block->size);
			prev_block->size += sizeof(t_block) + current_block->size;
			newIndex = compute_index(prev_block->size);

			set_prev_size(prev_block);

			if (oldIndex != newIndex)
			{
				remove_list_element(prev_block, oldIndex);
				add_list_element(prev_block, newIndex);
			}
			break;
		case NEXT_BLOCK_IS_FREE:
			remove_list_element(next_block, compute_index(next_block->size));
			current_block->size += sizeof(t_block) + next_block->size;
			set_prev_size(current_block);
			add_list_element(current_block, compute_index(current_block->size));
			break;
		default:
			add_list_element(current_block, compute_index(current_block->size));
			break;
	}
}

void* get_memory(uint64_t size)
{
	void* ptr;

	size = align_size(size);

	pthread_mutex_lock(&g_main_mutex);

	ptr = get_memory_unsafe(size);

	pthread_mutex_unlock(&g_main_mutex);

	return ptr;
}

void* calloc_memory(uint64_t nmemb, uint64_t size)
{
	void* new_ptr;

	size = align_size(nmemb * size);

	if (size == 0)
		return NULL;

	pthread_mutex_lock(&g_main_mutex);

	new_ptr = get_memory_unsafe(size);
	memset(new_ptr, 0, size);

	pthread_mutex_unlock(&g_main_mutex);

	return new_ptr;
}

void* realloc_memory(void* ptr, uint64_t size)
{
	void*		new_ptr;
	t_block*	current_block;

	if (size == 0)
	{
		free_memory(ptr);
		return NULL;
	}

	size = align_size(size);

	pthread_mutex_lock(&g_main_mutex);

	//If the pointer doesn't come from a get_memory()
	if (g_memory_head == NULL || ptr <= g_memory_head || (uint64_t)g_memory_head + g_memory_size < (uint64_t)ptr)
	{
		new_ptr = get_memory_unsafe(size);

		pthread_mutex_unlock(&g_main_mutex);

		return new_ptr;
	}

	current_block = (t_block*)((uint64_t)ptr - sizeof(t_block));

	//Instant split if we ask a smaller size
	if (current_block->size > sizeof(t_block) + size)
	{
		split_memory(current_block, size);

		pthread_mutex_unlock(&g_main_mutex);

		return ptr;
	}
	else if (current_block->size >= size) //That means that we have not the place for block datas
	{
		pthread_mutex_unlock(&g_main_mutex);

		return ptr;
	}

	//New pointer for a bigger size
	new_ptr = get_memory_unsafe(size);

	//We copy old datas at the new address
	if (size > current_block->size)
		memcpy(new_ptr, ptr, current_block->size);
	else
		memcpy(new_ptr, ptr, size);

	//Free old pointer
	free_memory_unsafe(ptr);

	pthread_mutex_unlock(&g_main_mutex);

	return new_ptr;
}

void free_memory(void* ptr)
{
	pthread_mutex_lock(&g_main_mutex);

	free_memory_unsafe(ptr);

	pthread_mutex_unlock(&g_main_mutex);
}

//It will probably never be used because the program free its memory himself when it get closed
void release_memory()
{
	if (g_memory_head == NULL)
		return;

	if (munmap(g_memory_head, g_memory_size) != 0)
		fprintf(stderr, "Error in release_memory: munmap failed\n");
}

void display_memory()
{
	uint64_t		allocated_size = 0;
	uint64_t		freed_size = 0;
	t_block*	block = g_memory_head;

	if (g_memory_head == NULL)
		return;

	while ((uint64_t)block < (uint64_t)g_memory_head + g_memory_offset)
	{
		if (block->next_free == g_impossible_address)
			allocated_size += block->size;
		else
			freed_size += block->size;

		block = (t_block*)((uint64_t)block + sizeof(t_block) + block->size);
	}

	printf("Allocated bytes : %lu\n", allocated_size);
	printf("Freed bytes : %lu\n", freed_size);
	printf("Unused bytes : %lu\n", g_memory_size - allocated_size - freed_size);
}
