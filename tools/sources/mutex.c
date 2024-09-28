#include <stdio.h>
#include <stdlib.h>

#include "mutex.h"

#ifdef  linux

void mutex_init(MUTEX* mutex)
{
	if (pthread_mutex_init(mutex, NULL) != 0)
	{
		fprintf(stderr, "Error in mutex_init: cannot create a new mutex\n");
		exit(1);
	}
}

void mutex_lock(MUTEX* mutex)
{
	pthread_mutex_lock(mutex);
}

void mutex_unlock(MUTEX* mutex)
{
	pthread_mutex_unlock(mutex);
}

void mutex_destroy(MUTEX* mutex)
{
	pthread_mutex_destroy(mutex);
}

#else

#include <Windows.h>

void mutex_init(MUTEX* mutex)
{
	*mutex = CreateMutex(NULL, FALSE, NULL);
	if (*mutex == NULL)
	{
		fprintf(stderr, "Error in mutex_init: cannot create a new mutex\n");
		exit(1);
	}
}

void mutex_lock(MUTEX* mutex)
{
	WaitForSingleObject(*mutex, INFINITE);
}

void mutex_unlock(MUTEX* mutex)
{
	ReleaseMutex(*mutex);
}

void mutex_destroy(MUTEX* mutex)
{
	CloseHandle(*mutex);
}

#endif