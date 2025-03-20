#include <chrono>
#include <iostream>
#include <thread>

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
				{
					HandleHandshake(*reinterpret_cast<HandshakeMessage*>(message.get()));
					break;
				}
				case MessageType::ERROR:
				{
					HandleError(*reinterpret_cast<ErrorMessage*>(message.get()));
					break;
				}
				default:
					std::cerr << "Unkown message type : " << static_cast<uint32_t>(message->GetType()) << std::endl;
			}
		}
	}
}

void Worker::HandleHandshake(const HandshakeMessage& handshake)
{
	std::cout << handshake.GetUniqueID() << std::endl;
	if (handshake.GetUniqueID() == 0)
	{
		m_handshake_is_running = false;
		std::cout << "Connection is initialized" << std::endl;
	}
	else if (!m_handshake_is_running)
	{
		std::thread handshake_thread(&Worker::HandshakeLoop, this, handshake.GetUniqueID());
		handshake_thread.detach();
		m_handshake_is_running = true;
	}
}

void Worker::HandshakeLoop(uint32_t unique_id) noexcept
{
	while (m_handshake_is_running)
	{
		m_client->SendDirectMessage(HandshakeMessage{ unique_id });
		std::this_thread::sleep_for(std::chrono::milliseconds(250));
	}
}

void Worker::HandleError(const ErrorMessage& error)
{
	switch (error.GetError())
	{
		case ErrorType::LOGIN_FAILED:
			std::cerr << "Login failed" << std::endl;
			break;
		default:
			std::cerr << "Unkown error type : " << static_cast<uint32_t>(error.GetError()) << std::endl;
	}
}
