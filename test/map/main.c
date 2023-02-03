#include <sys/time.h>
#include <time.h>
#include <sys/resource.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unordered_map>

#include "map.h"
#include "single_memory.h"

#define TEST_SIZE 100000

static long int get_timestamp_microsecond()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}

int main()
{
    int                             i;
    std::unordered_map<int, int*>   map;
    t_map                           hash_map;
    int                             arr[TEST_SIZE];
    int                             check[TEST_SIZE];
    long int	                    before;
    long int	                    duration;

    init_map(&hash_map);
    
    srand(time(NULL));
    for (i = 0; i < TEST_SIZE; ++i)
    {
        arr[i] = rand();
        check[i] = arr[i];
    }

    printf("-----------------------------------------------------------------------------\n");
    printf("UNORDERED_MAP (NOT THREAD SAFE):\n");
    before = get_timestamp_microsecond();
    for (i = 0; i < TEST_SIZE; ++i)
    {
        map[arr[i]] = &arr[i];
    }
    duration = get_timestamp_microsecond() - before;
    printf("%d add key/value iterations:\ttime elapsed = %ld microseconds\n", TEST_SIZE, duration);

    before = get_timestamp_microsecond();
    for (i = 0; i < TEST_SIZE; ++i)
    {
        if (check[i] != *map[arr[i]])
            printf("Error, wrong value returned by unordered_map\n");
    }
    duration = get_timestamp_microsecond() - before;
    printf("%d get value iterations:\t\ttime elapsed = %ld microseconds\n", TEST_SIZE, duration);

    before = get_timestamp_microsecond();
    for (i = 0; i < TEST_SIZE; ++i)
    {
        map.erase(arr[i]);
    }
    duration = get_timestamp_microsecond() - before;
    printf("%d remove value iterations:\t\ttime elapsed = %ld microseconds\n", TEST_SIZE, duration);

    for (i = 0; i < TEST_SIZE; ++i)
    {
        map[arr[i]] = &arr[i];
    }
    before = get_timestamp_microsecond();
    map.clear();
    duration = get_timestamp_microsecond() - before;
    printf("clear everything:\t\t\ttime elapsed = %ld microseconds\n", duration);

    printf("-----------------------------------------------------------------------------\n");
    printf("HASH_MAP (THREAD SAFE):\n");
    before = get_timestamp_microsecond();
    for (i = 0; i < TEST_SIZE; ++i)
    {
        add_map_element(&hash_map, arr[i], &arr[i]);
    }
    duration = get_timestamp_microsecond() - before;
    printf("%d add key/value iterations:\ttime elapsed = %ld microseconds\n", TEST_SIZE, duration);

    before = get_timestamp_microsecond();
    for (i = 0; i < TEST_SIZE; ++i)
    {
        if (check[i] != *(int*)get_map_element(&hash_map, arr[i]))
            printf("Error, wrong value returned by hash_map\n");
    }
    duration = get_timestamp_microsecond() - before;
    printf("%d get value iterations:\t\ttime elapsed = %ld microseconds\n", TEST_SIZE, duration);

    before = get_timestamp_microsecond();
    for (i = 0; i < TEST_SIZE; ++i)
    {
        remove_map_element(&hash_map, arr[i]);
    }
    duration = get_timestamp_microsecond() - before;
    printf("%d remove value iterations:\t\ttime elapsed = %ld microseconds\n", TEST_SIZE, duration);

    for (i = 0; i < TEST_SIZE; ++i)
    {
        add_map_element(&hash_map, arr[i], &arr[i]);
    }
    before = get_timestamp_microsecond();
    delete_map(&hash_map);
    duration = get_timestamp_microsecond() - before;
    printf("clear everything:\t\t\ttime elapsed = %ld microseconds\n", duration);
}