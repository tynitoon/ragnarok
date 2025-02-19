#include <iostream>
#include <thread>

#include "server.hpp"

int main() {
	std::shared_ptr<Server> server = std::make_shared<Server>(4242);
	std::thread server_thread(&Server::Run, server);

	Sleep(10000);

	for (int i = 0; i < 10000; ++i)
	{
		server->SendMessage(0, HandshakeMessage{1});
	}

	server_thread.join();

	return 0;
}
