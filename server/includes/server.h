#ifndef SERVER_H
#define SERVER_H

#include "list.h"
#include "message.h"

typedef struct
{
	int			sockfd;
	t_message	message;
}				t_dest_message;

/*!
 * \brief start a TCP server
 *
 * \param[out] messages_received by clients (list of t_dest_message)
 * \param[in] messages_to_send to clients (list of list of t_dest_message, each list is for a specific sockfd)
 */
void server_start(int port, t_list* messages_received, t_list* messages_to_send);

#endif
