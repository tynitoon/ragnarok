#ifndef CLIENT_H
#define CLIENT_H

#include "list.h"

/*!
 * \brief start a TCP client
 *
 * \param[out] messages received by the server (list of t_message)
 * \param[int] messages to send to the server (list of t_message)
 */
void client_start(char* ip, int port, t_list* messages_received, t_list* messages_to_send);

#endif
