#ifndef SINGLE_MEMORY_H
#define SINGLE_MEMORY_H

#include <stdint.h>

/* Used to easily replace allocation functions */
#define MALLOC(size)			get_memory(size)
#define CALLOC(number, size)	calloc_memory(number, size)
#define REALLOC(ptr, size)		realloc_memory(ptr, size)
#define FREE(ptr)				free_memory(ptr)

/*
 * /brief find the best fit in 1Go memory (allocated at the first call) and return an unused block having a size in bytes
 * 
 * /param[in] size of the asked block in byte
 *
 * /return the address of a free block having the asked size
 */
void* get_memory(uint64_t size);

/*
 * /brief find the best fit in 1Go memory (allocated at the first call) and return an unused block having a size of nmemb * size bytes
 * 
 * /param[in] nmemb is a number of element 
 * /param[in] size in byte of one element
 *
 * /return the address of a free block having a size of nmemb * size bytes
 */
void* calloc_memory(uint64_t nmemb, uint64_t size);

/*
 * /brief used to resize and already allocated block. If ptr is NULL it uses get_memory
 * 
 * /param[in] ptr is the address of the allocated block
 * /param[in] size is the new size in byte of the block
 *
 * /return the address of a free block having the asked size.
 */
void* realloc_memory(void* ptr, uint64_t size);

/*
 * /brief Release the memory to be able to be reused later
 *
 * /param[in] ptr is the address of the allocated block
 */
void free_memory(void* ptr);

/*
 * /brief display allocated, freed and unused memory
 */
void display_memory();

#endif /* SINGLE_MEMORY_H */