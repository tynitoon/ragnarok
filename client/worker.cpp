#include <iostream>

#include "worker.hpp"

Worker::Worker(const std::shared_ptr<Client>& client) noexcept :
	m_client(client),
	m_handshake_is_running(false)
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
				case MessageType::HANDSHAKE:
					Handshake(*reinterpret_cast<HandshakeMessage*>(message.get()));
					break;
				default:
					std::cerr << "Unkown message type : " << static_cast<uint32_t>(message->GetType()) << std::endl;
			}
		}
	}
}

void Worker::Handshake(HandshakeMessage& message)
{
	if (message.GetUniqueID() == 0)
	{
		m_handshake_is_running = false;
		std::cout << "Connection is initialized" << std::endl;
	}
	else if (!m_handshake_is_running)
	{
		try
		{
			std::thread handshake_thread(&Worker::HandshakeLoop, this, message.GetUniqueID());
			handshake_thread.detach();
			m_handshake_is_running = true;
		}
		catch (std::exception e)
		{
			std::cerr << e.what() << std::endl;
		}
	}
}

void Worker::HandshakeLoop(uint32_t unique_id) noexcept
{
	while (m_handshake_is_running)
	{
		m_client->SendDirectMessage(HandshakeMessage{ unique_id });
		Sleep(250);
	}
}
