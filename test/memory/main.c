#include <sys/time.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>

#include "single_memory.h"

#define TEST_SIZE 100000

long int get_timestamp_microsecond()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000000 + tv.tv_usec;
}

int main()
{
	int			i;
	int			j;
	char*		ptr[TEST_SIZE];
	int			size[TEST_SIZE];
	int			random_nb[TEST_SIZE];
	long int	before;
	long int	duration;

	srand(time(NULL));
	for (i = 0; i < TEST_SIZE; ++i)
	{
		size[i] = rand() % 5 * sizeof(int);
		random_nb[i] = rand();
	}

	printf("-----------------------------------------------------------------------------\n");
	printf("MALLOC:\n");
	ptr[0] = malloc(sizeof(char)); //Init memory
	before = get_timestamp_microsecond();
	for (i = 0; i < TEST_SIZE; ++i)
	{
		ptr[i] = malloc(sizeof(char));
	}
	duration = get_timestamp_microsecond() - before;
	printf("%d malloc() iterations:\t\ttime elapsed = %ld microseconds\n", TEST_SIZE, duration);

	before = get_timestamp_microsecond();
	for (i = 0; i < TEST_SIZE; ++i)
	{
		ptr[i] = realloc(ptr[i], size[i]);
	}
	duration = get_timestamp_microsecond() - before;
	printf("%d realloc() iterations:\t\ttime elapsed = %ld microseconds\n", TEST_SIZE, duration);

	before = get_timestamp_microsecond();
	for (i = 0; i < TEST_SIZE; ++i)
	{
		free(ptr[i]);
	}
	duration = get_timestamp_microsecond() - before;
	printf("%d free() iterations:\t\ttime elapsed = %ld microseconds\n", TEST_SIZE, duration);


	j = 0;
	before = get_timestamp_microsecond();
	for (i = 0; i < TEST_SIZE; ++i)
	{
		if (random_nb[i] % 2 == 0)
		{
			ptr[j] = malloc(size[i]);
			++j;
		}
		else if (random_nb[i] % 3 == 0 && j > 0)
			ptr[j - 1] = realloc(ptr[j - 1], size[i]);
		else if (j > 0)
		{
			j--;
			free(ptr[j]);
			ptr[j] = NULL;
		}
	}
	duration = get_timestamp_microsecond() - before;
	printf("%d mixed iterations:\t\ttime elapsed = %ld microseconds\n", TEST_SIZE, duration);

	printf("-----------------------------------------------------------------------------\n");
	printf("SINGLE_MEMORY:\n");
	ptr[0] = get_memory(sizeof(char)); //Init memory
	before = get_timestamp_microsecond();
	for (i = 0; i < TEST_SIZE; ++i)
	{
		ptr[i] = get_memory(sizeof(char));
	}
	duration = get_timestamp_microsecond() - before;
	printf("%d get_memory() iterations:\t\ttime elapsed = %ld microseconds\n", TEST_SIZE, duration);

	before = get_timestamp_microsecond();
	for (i = 0; i < TEST_SIZE; ++i)
	{
		ptr[i] = realloc_memory(ptr[i], size[i]);
	}
	duration = get_timestamp_microsecond() - before;
	printf("%d realloc_memory() iterations:\ttime elapsed = %ld microseconds\n", TEST_SIZE, duration);

	before = get_timestamp_microsecond();
	for (i = 0; i < TEST_SIZE; ++i)
	{
		free_memory(ptr[i]);
	}
	duration = get_timestamp_microsecond() - before;
	printf("%d free_memory() iterations:\ttime elapsed = %ld microseconds\n", TEST_SIZE, duration);

	j = 0;
	before = get_timestamp_microsecond();
	for (i = 0; i < TEST_SIZE; ++i)
	{
		if (random_nb[i] % 2 == 0)
		{
			ptr[j] = get_memory(size[i]);
			++j;
		}
		else if (random_nb[i] % 3 == 0 && j > 0)
			ptr[j - 1] = realloc_memory(ptr[j - 1], size[i]);
		else if (j > 0)
		{
			j--;
			free_memory(ptr[j]);
			ptr[j] = NULL;
		}
	}
	duration = get_timestamp_microsecond() - before;
	printf("%d mixed iterations:\t\ttime elapsed = %ld microseconds\n", TEST_SIZE, duration);
	printf("-----------------------------------------------------------------------------\n");

	return 0;
}