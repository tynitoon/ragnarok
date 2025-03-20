#include <iostream>

#include "worker.hpp"

Worker::Worker(const std::shared_ptr<Server>& server) noexcept :
	m_server(server)
{}

void Worker::Run()
{
	while (true)
	{
		std::unique_ptr<MessageFrom> message = m_server->ReadMessage();
		if (message.get() != nullptr)
		{
			switch (message->message->GetType())
			{
				case MessageType::LOGIN:
				{
					//LoginMessage* login = reinterpret_cast<LoginMessage*>(message->message.get());
					//std::cout << "Worker::Run: Login message received : " << login->GetUsername() << " " << login->GetPassword() << std::endl;
					break;
				}
				default:
					break;
			}
		}
	}
}
