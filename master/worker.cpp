#include <iostream>

#include "worker.hpp"

Worker::Worker(const std::shared_ptr<Server>& server) noexcept :
	m_server(server),
	m_database("127.0.0.1")
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
				HandleLogin(message->fd, *reinterpret_cast<LoginMessage*>(message->message.get()));
				break;
			}
			default:
				break;
			}
		}
	}
}

void Worker::HandleLogin(uint32_t fd, const LoginMessage& login)
{
	std::cout << "Worker::Login: Login message received : " << login.GetUsername() << " " << login.GetPassword() << std::endl;
	if (m_database.CheckLogin(login.GetUsername(), login.GetPassword()))
		std::cout << "Worker::Login: Login success" << std::endl;
	else
	{
		std::cout << "Worker::Login: Login failed" << std::endl;
		m_server->SendMessage(fd, ErrorMessage(ErrorType::LOGIN_FAILED, "Login failed"));
	}
}
