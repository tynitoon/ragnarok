#include <stddef.h>

#define GET_16_BITS(value) ((unsigned int)(*(unsigned short*)(value)))

unsigned int generate_hash(void* data, unsigned int length)
{
    unsigned int    hash;
    int             remaining_bits;

    if (data == NULL)
        return 0;

    hash = length;
    remaining_bits = length & 3;
    length >>= 2;

    while (length > 0)
    {
        hash += GET_16_BITS(data);
        hash = (hash << 16) ^ ((GET_16_BITS(data + 2) << 11) ^ hash);
        data += sizeof(unsigned int);
        hash += hash >> 11;

        --length;
    }

    switch (remaining_bits)
    {
    case 3:
        hash += GET_16_BITS(data);
        hash ^= hash << 16;
        hash ^= ((signed char)((char*)data)[sizeof(unsigned short)]) << 18;
        hash += hash >> 11;
        break;
    case 2:
        hash += GET_16_BITS(data);
        hash ^= hash << 11;
        hash += hash >> 17;
        break;
    case 1:
        hash += (signed char)*((char*)data);
        hash ^= hash << 10;
        hash += hash >> 1;
    }

    hash ^= hash << 3;
    hash += hash >> 5;
    hash ^= hash << 4;
    hash += hash >> 17;
    hash ^= hash << 25;
    hash += hash >> 6;

    return hash;
}