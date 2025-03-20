#include "client.hpp"
#include "worker.hpp"
#include "hasher.hpp"

int main() {
	std::shared_ptr<Client> client = std::make_shared<Client>("127.0.0.1", 4242, 4243);
	std::vector<std::thread> threads;
	threads.push_back(std::thread([&client]()
		{
			Worker worker(client);
			worker.Run();
		}));
	threads.push_back(std::thread(&Client::Run, client));

	Sleep(2000);

	Hasher hasher;
	client->SendMessage(LoginMessage("tynitoon", hasher.sha256("azerty")));

	//TODO : game->Run();

	for (auto& thread : threads)
		thread.join();

	return 0;
}