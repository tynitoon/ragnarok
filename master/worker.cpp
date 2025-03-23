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
	int key = m_database.CheckLogin(login.GetUsername(), login.GetPassword());
	if (key == -1)
	{
		std::cout << "Worker::Login: Login failed" << std::endl;
		m_server->SendMessage(fd, ErrorMessage(ErrorType::LOGIN_FAILED, "Login failed"));
		return;
	}

	const std::vector server_keys = m_database.AccountKeyToServerKeys(key);
	// find a way to link server_keys with the unique ID that is stored in the server object
	// then send message to make servers disconnect the client

	std::string auth_key = m_hasher.sha256(std::string(login.GetUsername()) + std::to_string(std::time(0)));
	
	// generate auth_key and push it in data base
	// send auth_key to client with list of its characters
	// client send to the master server : character key
	// master server send to the client : ip, port, character key, auth_key
	// client send to the game server : character key, auth_key
}

//void Worker::HandleSelectCharacter(uint32_t fd, const LoginMessage& login) // -> to define
//{
//	// verify that character exist and is linked to the account
//	// if a server is available, if not create a server connect to the server add it to database
//	// link the character to the server in the database
//	// send the server ip and port to the client
//}