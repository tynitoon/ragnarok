#include "server.hpp"

int main() {
	int port = 4242;
	Server server(port);
	server.Run();

	return 0;
}
