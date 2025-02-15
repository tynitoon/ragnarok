#ifndef MUTEX_H
#define MUTEX_H

#ifdef linux
#include <pthread.h>

typedef pthread_mutex_t	MUTEX;
#else
typedef void*			MUTEX;
#endif /* Linux or windows */

/*!
 * /brief init a mutex
 *
 * /param[in] mutex is the mutex to init
 */
void mutex_init(MUTEX* mutex);

/*!
 * /brief lock a mutex
 *
 * /param[in] mutex is the mutex to lock
 */
void mutex_lock(MUTEX* mutex);

/*!
 * /brief unlock a mutex
 *
 * /param[in] mutex is the mutex to unlock
 */
void mutex_unlock(MUTEX* mutex);

/*!
 * /brief destroy a mutex
 *
 * /param[in] mutex is the mutex to destroy
 */
void mutex_destroy(MUTEX* mutex);

#endif /* MUTEX_H */