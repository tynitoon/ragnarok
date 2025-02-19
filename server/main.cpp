#include "server.hpp"
#include "worker.hpp"

int main()
{
	/* Compute the number of worker that we can have */
	std::size_t nb_worker = std::min(static_cast<int>(std::thread::hardware_concurrency() - 1), 1);

	/* Create worker threads */
	std::vector<std::thread> threads;
	std::shared_ptr<Server> server = std::make_shared<Server>(4242);
	for (std::size_t i = 0; i < nb_worker; ++i)
	{
		threads.push_back(std::thread([&server]()
			{
				Worker worker(server);
				worker.Run();
			}));
	}

	server->Run();

	for (auto& thread : threads)
		thread.join();

	return 0;
}
