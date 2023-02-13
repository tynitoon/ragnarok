#ifndef SINGLE_MEMORY_H
#define SINGLE_MEMORY_H

#include <stdint.h>

void*	get_memory(uint64_t size);
void*	calloc_memory(uint64_t nmemb, uint64_t size);
void*	realloc_memory(void* ptr, uint64_t size);
void	free_memory(void* ptr);
void	display_memory();

#endif