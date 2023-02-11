//#include <stdio.h>
//#include <string.h>
//#include <errno.h>
//#include <unistd.h>
//#include <arpa/inet.h>
//#include <sys/types.h>
//#include <sys/socket.h>
//#include <netinet/in.h>
//#include <sys/time.h>
//
//#include "list.h"
//#include "protocol.h"
//#include "client.h"
//#include "single_memory.h"
//
//int start_client(char* address, int port, t_server* server)
//{
//	int 				ret_value;
//	struct sockaddr_in  server_address;
//	t_list_element*		save;
//	t_message*          message;
//	size_t				message_size;
//
//	//Create socket
//	if ((server->fd = socket(AF_INET, SOCK_STREAM, 0)) == -1)
//	{
//		perror("socket failed");
//		return -1;
//	}
//
//	memset(&server_address, 0, sizeof(server_address));
//
//	//Assign server ip and port
//	server_address.sin_family = AF_INET;
//	server_address.sin_addr.s_addr = inet_addr(address);
//	server_address.sin_port = htons(port);
//
//	//Connect to server
//	if (connect(server->fd, (struct sockaddr*)&server_address, sizeof(server_address)) != 0)
//	{
//		perror("connect failed");
//		return -1;
//	}
//
//	while (1)
//	{
//		//Check if it was for closing and read the incoming message
//		if ((ret_value = read(server->fd, &server->buffer[server->buffer_index], BUFFER_SIZE - server->buffer_index)) == 0)
//		{
//			//Close the socket
//			close(server->fd);
//			server->fd = -1;
//
//			return 0;
//		}
//		else
//		{
//			server->buffer_index += ret_value;
//
//			//If there is enough data to get the message's size
//			while (server->buffer_index >= (int)sizeof(int))
//			{
//				message = (t_message*)server->buffer;
//				message_size = message->size;
//
//				//if the message_size is impossible (cheater or network error)
//				if (message_size < sizeof(t_message) || message_size > BUFFER_SIZE)
//					server->buffer_index = 0;
//				else if (server->buffer_index >= message_size)
//				{
//					save = get_memory(sizeof(t_list_element) + message_size);
//					message = (t_message*)save->data;
//
//					//Copy datas
//					memcpy(message, server->buffer, message_size);
//
//					//Change the size to have only the data size
//					message->size = message_size - sizeof(t_message);
//					add_list_element(&server->messages, save);
//
//					//Shift datas in the buffer (circular buffer)
//					server->buffer_index -= message_size;
//					memmove(server->buffer, &server->buffer[message_size], server->buffer_index);
//				}
//				else
//					break;
//			}
//		}
//	}
//
//	//// function for chat
//	//message.type = CONNECT;
//	//message.size = sizeof(t_message) + sizeof(t_connect);
//	//printf("message.size = %lu %lu %lu\n", message.size, sizeof(t_message), sizeof(t_connect));
//	//memcpy(connect_message.username, "default\0", strlen("default") + 1);
//	//memcpy(connect_message.password, "default\0", strlen("default") + 1);
//
//	//memcpy(buffer, &message, sizeof(t_message));
//	//memcpy(&buffer[sizeof(t_message)], &connect_message, sizeof(t_connect));
//
//	//printf("user = %s password = %s\n", (char*)(&buffer[sizeof(t_message)]), (char*)(&buffer[sizeof(t_message) + 32]));
//
//	//if (write(fd, buffer, message.size) < 0)
//	//	return -1;
//
//	//// close the socket
//	//close(fd);
//
//	return 0;
//}

//#define WIN32_LEAN_AND_MEAN
//
//#include <windows.h>
//#include <winsock2.h>
//#include <ws2tcpip.h>
//#include <stdlib.h>
//#include <stdio.h>
//
//#include "client.h"
//
//// Need to link with Ws2_32.lib, Mswsock.lib, and Advapi32.lib
//#pragma comment (lib, "Ws2_32.lib")
//#pragma comment (lib, "Mswsock.lib")
//#pragma comment (lib, "AdvApi32.lib")
//
//
//#define DEFAULT_BUFLEN 512
//#define DEFAULT_PORT "27015"
//
//int start_client(char* address, int port, t_server* server);
//{
//    WSADATA wsaData;
//    SOCKET ConnectSocket = INVALID_SOCKET;
//    struct addrinfo* result = NULL,
//        * ptr = NULL,
//        hints;
//    const char* sendbuf = "this is a test";
//    char recvbuf[DEFAULT_BUFLEN];
//    int iResult;
//    int recvbuflen = DEFAULT_BUFLEN;
//
//    // Validate the parameters
//    if (argc != 2) {
//        printf("usage: %s server-name\n", argv[0]);
//        return 1;
//    }
//
//    // Initialize Winsock
//    iResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
//    if (iResult != 0) {
//        printf("WSAStartup failed with error: %d\n", iResult);
//        return 1;
//    }
//
//    ZeroMemory(&hints, sizeof(hints));
//    hints.ai_family = AF_UNSPEC;
//    hints.ai_socktype = SOCK_STREAM;
//    hints.ai_protocol = IPPROTO_TCP;
//
//    // Resolve the server address and port
//    iResult = getaddrinfo(argv[1], DEFAULT_PORT, &hints, &result);
//    if (iResult != 0) {
//        printf("getaddrinfo failed with error: %d\n", iResult);
//        WSACleanup();
//        return 1;
//    }
//
//    // Attempt to connect to an address until one succeeds
//    for (ptr = result; ptr != NULL; ptr = ptr->ai_next) {
//
//        // Create a SOCKET for connecting to server
//        ConnectSocket = socket(ptr->ai_family, ptr->ai_socktype,
//            ptr->ai_protocol);
//        if (ConnectSocket == INVALID_SOCKET) {
//            printf("socket failed with error: %ld\n", WSAGetLastError());
//            WSACleanup();
//            return 1;
//        }
//
//        // Connect to server.
//        iResult = connect(ConnectSocket, ptr->ai_addr, (int)ptr->ai_addrlen);
//        if (iResult == SOCKET_ERROR) {
//            closesocket(ConnectSocket);
//            ConnectSocket = INVALID_SOCKET;
//            continue;
//        }
//        break;
//    }
//
//    freeaddrinfo(result);
//
//    if (ConnectSocket == INVALID_SOCKET) {
//        printf("Unable to connect to server!\n");
//        WSACleanup();
//        return 1;
//    }
//
//    // Send an initial buffer
//    iResult = send(ConnectSocket, sendbuf, (int)strlen(sendbuf), 0);
//    if (iResult == SOCKET_ERROR) {
//        printf("send failed with error: %d\n", WSAGetLastError());
//        closesocket(ConnectSocket);
//        WSACleanup();
//        return 1;
//    }
//
//    printf("Bytes Sent: %ld\n", iResult);
//
//    // shutdown the connection since no more data will be sent
//    iResult = shutdown(ConnectSocket, SD_SEND);
//    if (iResult == SOCKET_ERROR) {
//        printf("shutdown failed with error: %d\n", WSAGetLastError());
//        closesocket(ConnectSocket);
//        WSACleanup();
//        return 1;
//    }
//
//    // Receive until the peer closes the connection
//    do {
//
//        iResult = recv(ConnectSocket, recvbuf, recvbuflen, 0);
//        if (iResult > 0)
//            printf("Bytes received: %d\n", iResult);
//        else if (iResult == 0)
//            printf("Connection closed\n");
//        else
//            printf("recv failed with error: %d\n", WSAGetLastError());
//
//    } while (iResult > 0);
//
//    // cleanup
//    closesocket(ConnectSocket);
//    WSACleanup();
//
//    return 0;
//}