#include <chrono>
#include <iostream>

#include "worker.hpp"

Worker::Worker(const std::shared_ptr<Client>& client) noexcept :
	m_client(client)
{}

void Worker::Run()
{
	while (true)
	{
		auto message = m_client->ReadMessage();
		if (message.get() != nullptr)
		{
			switch (message->GetType())
			{
				default:
					std::cerr << "Unkown message type : " << static_cast<uint32_t>(message->GetType()) << std::endl;
			}
		}
	}
}
