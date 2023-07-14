#ifndef HASH_H
#define HASH_H

/*
 * /brief hash data into an unsigned int
 *
 * /param[in] data that will be hashed
 * /param[in] length is the size in bit of the data
 *
 * /return an unsigned int which is the hash of the data
 */
unsigned int generate_hash(void* data, unsigned int length);

#endif
