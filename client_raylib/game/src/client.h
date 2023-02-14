#ifndef CLIENT_H
#define CLIENT_H

#include <stdint.h>
#include "protocol.h"
#include "list.h"

#define BUFFER_SIZE 4096

/* Define Things to avoid include windows.h */
#if defined(_WIN64)
typedef __int64 INT_PTR, * PINT_PTR;
typedef unsigned __int64 UINT_PTR, * PUINT_PTR;

typedef __int64 LONG_PTR, * PLONG_PTR;
typedef unsigned __int64 ULONG_PTR, * PULONG_PTR;

#define __int3264   __int64

#else
typedef _W64 int INT_PTR, * PINT_PTR;
typedef _W64 unsigned int UINT_PTR, * PUINT_PTR;

typedef _W64 long LONG_PTR, * PLONG_PTR;
typedef _W64 unsigned long ULONG_PTR, * PULONG_PTR;

#define __int3264   __int32

#endif

typedef UINT_PTR    SOCKET;
/* End of defines for windows.h */

typedef struct      s_server
{
    SOCKET          fd;
    t_list          messages;
    uint64_t        buffer_index;
    char            buffer[BUFFER_SIZE];
}                   t_server;

int start_client(char* address, char* port, t_server* server);

#endif
