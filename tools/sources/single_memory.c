#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <inttypes.h> /* For PRIu64 */
#include <stdbool.h>

#include "mutex.h"
#include "single_memory.h"

#define PAGE_NUMBER				18									/*!< In this system a page is 4096 bytes, so 4096 * 2^18 = 1Go */
#define MAX_FREE_INDEX			40									/*!< Each index is a power of 2. So max size can be 2^40 (more than 1 000 000 000 000) */
#define NOT_INITIALIZED_VALUE	100									/*!< Used to init g_table_index so we just have to compare index with */

/*!
 * \brief flag used to determine which block is freed
 */
typedef enum
{
	PREV_BLOCK_IS_FREE	= (1 << 0),
	NEXT_BLOCK_IS_FREE	= (1 << 1),
	BOTH_BLOCK_ARE_FREE = PREV_BLOCK_IS_FREE | NEXT_BLOCK_IS_FREE
}								t_filter;

/*!
 * \brief memory block
 */
typedef struct					s_block
{
	uint64_t					size;								/*!< Navigate to the next block */
	uint64_t					prev_size;							/*!< Navigate to the previous block */
	struct s_block*				next_free;							/*!< Double linked list on free blocks. Equal g_impossible_address if not free or something else if it is */
	struct s_block*				prev_free;							/*!< Double linked list on free blocks */
}								t_block;

static uint64_t					g_memory_size;						/*!< Size of the allocation */
static uint64_t					g_memory_offset = 0;				/*!< Max address offset, if it reachs g_memory_size, the memory can be full if no free block match */
static void*					g_memory_head = NULL;				/*!< Address of the head memory (given at the first call by malloc) */
static t_block*					g_frees[MAX_FREE_INDEX];			/*!< Free blocks are ordered in it, depending of the compute_index(block->size)  */
static int						g_table_index[MAX_FREE_INDEX];		/*!< A index in free blocks can be empty. We use this table to reduce iterations */
static MUTEX					g_main_mutex;						/*!< For Thread safe allocation */
static const void*				g_impossible_address;				/*!< Impossible address, it used to detect a free element */

#ifdef  linux

#include <unistd.h>

/*!
 * \brief Find memory size to allocate based on page size
 *
 * \return the memory size to allocate in bytes
 */
static uint64_t find_memory_size()
{
	return (uint64_t)sysconf(_SC_PAGE_SIZE) << PAGE_NUMBER;
}

#else

#include <Windows.h>

/*!
 * \brief Find memory size to allocate based on page size
 *
 * \return the memory size to allocate in bytes
 */
static uint64_t find_memory_size()
{
	SYSTEM_INFO	system_infos;
	GetSystemInfo(&system_infos);
	return (uint64_t)system_infos.dwPageSize << PAGE_NUMBER;
}

#endif

/*!
 * \brief Quick check to determine an address is in the allocated memory
 *
 * \return true if the address is in the memory range, false otherwise
 */
static bool is_in_memory(void* ptr)
{
	return g_memory_head != NULL && g_memory_head < ptr && (uint64_t)ptr < (uint64_t)g_memory_head + g_memory_size;
}

/*!
 * \brief Quick Log2, it uses the Debruijn algorithm
 *
 * \param[in] nb is the number that will be transforms into its Log2 
 *
 * \return Log2(nb)
 */
static int compute_index(uint64_t nb)
{
	static const int	tab64[64] =
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

	nb |= nb >> 1;
	nb |= nb >> 2;
	nb |= nb >> 4;
	nb |= nb >> 8;
	nb |= nb >> 16;
	nb |= nb >> 32;


	return tab64[((uint64_t)((nb - (nb >> 1)) * 0x07EDD5E59A4E28C2)) >> 58];
}

/*!
 * \brief Align value to the current architecture to gain performances
 *
 * \param[in] size that needs to be aligned
 *
 * \return aligned size
 * \remark negative sizeof is used as mask, example with size == 5 : (5 + 8 - 1) & -8 => 0000 1100 & 1111 1000 => 0000 1000 == 8
 *         A negative number is full of 1, this integer 0b11111111111111111111111111111000 is equal to -8 and 0b11111111111111111111111111111111 is equal to -1
 */
static uint64_t align_size(uint64_t size)
{
	return (size + sizeof(void*) - 1) & -sizeof(void*);
}

/*!
 * \brief Insert list element at a specific free index
 *
 * \param[in] to_add is the block to add
 * \param[in] index of the g_free where we want to insert our element
 */
static void add_list_element(t_block* to_add, int index)
{
	/* Push front */
	to_add->prev_free = NULL;
	to_add->next_free = g_frees[index];

	if (g_frees[index] != NULL)
		g_frees[index]->prev_free = to_add;

	g_frees[index] = to_add;

	/* Update g_table_index */
	for (int i = 0; i <= index; ++i)
	{
		if (index < g_table_index[i])
			g_table_index[i] = index;
	}
}

/*!
 * \brief Remove list element from a specific free index
 *
 * \param[in] to_remove is the block to remove
 * \param[in] index of the g_free where we want to remove our element
 */
static void remove_list_element(t_block* to_remove, int index)
{
	if (to_remove == NULL)
		return;

	if (g_frees[index] == to_remove)
	{
		g_frees[index] = to_remove->next_free;

		/* Update g_table_index */
		if (g_frees[index] == NULL)
		{
			int	max_index = g_table_index[index + 1];
			for (int i = 0; i <= index; ++i)
			{
				if (g_table_index[i] == index)
					g_table_index[i] = max_index;
			}
		}
	}
	else
	{
		/* The variable prev_free is only used to remove a block from list without iteration */
		to_remove->prev_free->next_free = to_remove->next_free;

		if (to_remove->next_free != NULL)
			to_remove->next_free->prev_free = to_remove->prev_free;
	}

	to_remove->next_free = (t_block*)g_impossible_address;
}

/*!
 * \brief Set the size of the block in the prev_size of the next block
 *
 * \param[in] block that have a size that we want to write in the next block
 *
 * \remark We avoid problems by setting the prev_size variable before creating the a block
 */
static void set_prev_size(t_block* block)
{
	uint64_t	size = block->size;

	block = (t_block*)((uint64_t)block + sizeof(t_block) + size);
	if ((uint64_t)block < (uint64_t)g_memory_head + g_memory_size)
		block->prev_size = size;
}

/*!
 * \brief Split a block
 *
 * \param[in] block that we want to split
 * \param[in] size is the new size that we want for the block
 *
 * \remark With a split, the get_memory is a kind of best fit
 */
static void split_memory(t_block* block, uint64_t size)
{
	if (block->next_free != g_impossible_address)
		remove_list_element(block, compute_index(block->size));

	t_block* new_block = (t_block*)((uint64_t)block + sizeof(t_block) + size);
	new_block->size = block->size - sizeof(t_block) - size;
	new_block->prev_size = size;

	set_prev_size(new_block);

	add_list_element(new_block, compute_index(new_block->size));

	block->size = size;
}

/*!
 * \brief memory_get without mutex
 *
 * \param[in] size that we want to allocate
 */
static void* memory_get_unsafe(uint64_t size)
{
	if (size == 0)
		return NULL;

	t_block* new_block;
	/* Init memory if it is the first call */
	if (g_memory_head == NULL)
	{
		g_memory_size = find_memory_size();
		g_memory_head = malloc(g_memory_size);

		if (g_memory_head == NULL)
		{
			fprintf(stderr, "Error in get_memory: malloc failed\n");
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

		/* return the point on the data, not on the block */
		return (void*)((uint64_t)new_block + sizeof(t_block));
	}

	/* Avoid impossible values */
	if (size > g_memory_size - sizeof(t_block))
		return NULL;

	/* Search in free list */
	int	index = g_table_index[compute_index(size) + 1];
	if (index != NOT_INITIALIZED_VALUE)
	{
		t_block* block = g_frees[index];
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

	/* There is no place in free list. We have to create a new block using the offset */
	new_block = (t_block*)((uint64_t)g_memory_head + g_memory_offset);

	if ((uint64_t)new_block + sizeof(t_block) + size > (uint64_t)g_memory_head + g_memory_size)
	{
		fprintf(stderr, "Error in get_memory: memory is full or asked size is too big\n");
		return NULL;
	}

	new_block->size = size;
	new_block->next_free = (t_block*)g_impossible_address;

	set_prev_size(new_block);

	/* We update the offset */
	g_memory_offset += sizeof(t_block) + size;

	return (void*)((uint64_t)new_block + sizeof(t_block));
}

/*!
 * \brief memory_free without mutex
 *
 * \param[in] ptr that we want to free
 */
static void memory_free_unsafe(void* ptr)
{
	if (!is_in_memory(ptr))
		return;

	/* Init filter */
	int filter = 0;

	/* Check if previous block exists and get it if it does */
	t_block* prev_block = NULL;
	t_block* current_block = (t_block*)((uint64_t)ptr - sizeof(t_block));
	if (current_block->prev_size != 0)
	{
		prev_block = (t_block*)((uint64_t)current_block - sizeof(t_block) - current_block->prev_size);
		if (prev_block->next_free != g_impossible_address)
			filter |= PREV_BLOCK_IS_FREE;
	}

	/* Check if the next block is inside the allocated memory and get it if it is */
	t_block* next_block = NULL;
	if ((uint64_t)current_block + sizeof(t_block) + current_block->size < (uint64_t)g_memory_head + g_memory_offset)
	{
		next_block = (t_block*)((uint64_t)current_block + sizeof(t_block) + current_block->size);
		if (next_block->next_free != g_impossible_address)
			filter |= NEXT_BLOCK_IS_FREE;
	}

	/* Try to merge blocks */
	switch (filter)
	{
		case BOTH_BLOCK_ARE_FREE:
		{
			int oldIndex = compute_index(prev_block->size);
			prev_block->size += (sizeof(t_block) << 1) + current_block->size + next_block->size;
			int newIndex = compute_index(prev_block->size);

			set_prev_size(prev_block);

			remove_list_element(next_block, compute_index(next_block->size));

			if (oldIndex != newIndex)
			{
				remove_list_element(prev_block, oldIndex);
				add_list_element(prev_block, newIndex);
			}
			break;
		}
		case PREV_BLOCK_IS_FREE:
		{
			int oldIndex = compute_index(prev_block->size);
			prev_block->size += sizeof(t_block) + current_block->size;
			int newIndex = compute_index(prev_block->size);

			set_prev_size(prev_block);

			if (oldIndex != newIndex)
			{
				remove_list_element(prev_block, oldIndex);
				add_list_element(prev_block, newIndex);
			}
			break;
		}
		case NEXT_BLOCK_IS_FREE:
		{
			remove_list_element(next_block, compute_index(next_block->size));
			current_block->size += sizeof(t_block) + next_block->size;
			set_prev_size(current_block);
			add_list_element(current_block, compute_index(current_block->size));
			break;
		}
		default:
		{
			add_list_element(current_block, compute_index(current_block->size));
			break;
		}
	}
}

/*!
 * \brief Initialize the mutex
 */
static void init_mutex()
{
	if (g_memory_head == NULL)
		mutex_init(&g_main_mutex);
}

void* memory_get(uint64_t size)
{
	init_mutex();

	size = align_size(size);

	mutex_lock(&g_main_mutex);

	void* ptr = memory_get_unsafe(size);

	mutex_unlock(&g_main_mutex);

	return ptr;
}

void* memory_calloc(uint64_t nmemb, uint64_t size)
{
	init_mutex();

	size = align_size(nmemb * size);

	if (size == 0)
		return NULL;

	mutex_lock(&g_main_mutex);

	void* new_ptr = memory_get_unsafe(size);
	memset(new_ptr, 0, size);

	mutex_unlock(&g_main_mutex);

	return new_ptr;
}

void* memory_realloc(void* ptr, uint64_t size)
{
	init_mutex();

	if (size == 0)
	{
		memory_free(ptr);
		return NULL;
	}

	size = align_size(size);

	mutex_lock(&g_main_mutex);

	/* If the pointer doesn't come from a get_memory() */
	void* new_ptr;
	if (!is_in_memory(ptr))
	{
		new_ptr = memory_get_unsafe(size);

		mutex_unlock(&g_main_mutex);

		return new_ptr;
	}

	t_block* current_block = (t_block*)((uint64_t)ptr - sizeof(t_block));

	/* Instant split if we ask a smaller size */
	if (current_block->size > sizeof(t_block) + size)
	{
		split_memory(current_block, size);

		mutex_unlock(&g_main_mutex);

		return ptr;
	}
	else if (current_block->size >= size) /* That means that we have not the place for block datas and that we can't split */
	{
		mutex_unlock(&g_main_mutex);

		return ptr;
	}

	/* New pointer for a bigger size */
	new_ptr = memory_get_unsafe(size);

	/* We copy old datas at the new address */
	memcpy(new_ptr, ptr, current_block->size);

	/* Free old pointer */
	memory_free_unsafe(ptr);

	mutex_unlock(&g_main_mutex);

	return new_ptr;
}

void memory_free(void* ptr)
{
	init_mutex();

	mutex_lock(&g_main_mutex);

	memory_free_unsafe(ptr);

	mutex_unlock(&g_main_mutex);
}

void memory_display()
{
	init_mutex();

	mutex_lock(&g_main_mutex);

	if (g_memory_head == NULL)
	{
		printf("No memory have been allocated at the moment\n");
		mutex_unlock(&g_main_mutex);
		return;
	}

	uint64_t allocated_size = 0;
	uint64_t freed_size = 0;
	t_block* block = (t_block*)g_memory_head;
	while ((uint64_t)block < (uint64_t)g_memory_head + g_memory_offset)
	{
		if (block->next_free == g_impossible_address)
			allocated_size += block->size;
		else
			freed_size += block->size;

		printf("Address = %p prev size = %" PRIu64 " size = %" PRIu64 "\n", block, block->prev_size, block->size);

		block = (t_block*)((uint64_t)block + sizeof(t_block) + block->size);
	}

	printf("Allocated bytes : %" PRIu64 "\n", allocated_size);
	printf("Freed bytes : %" PRIu64 "\n", freed_size);
	printf("Unused bytes : %" PRIu64 "\n", g_memory_size - allocated_size - freed_size);

	mutex_unlock(&g_main_mutex);
}