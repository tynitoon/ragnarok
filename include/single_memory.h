#ifndef MEMORY_H
#define MEMORY_H

#include <stddef.h>

void* get_memory(size_t size);
void free_memory(void* ptr);
void release_memory();
void display_memory();

#endif