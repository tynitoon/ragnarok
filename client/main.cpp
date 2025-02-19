#include <iostream>
#include <thread>

#include "client.hpp"

int main() {
	std::shared_ptr<Client> client = std::make_shared<Client>("127.0.0.1", 4242);
	std::thread client_thread(&Client::Run, client);

	bool is_init = false;
	uint32_t unique_id = 0;
	while (!is_init)
	{
		auto message = client->ReadMessage();
		if (message.get() != nullptr)
		{
			std::cout << "Read it" << std::endl;
			switch (message->GetType())
			{
				case MessageType::HANDSHAKE:
				{
					HandshakeMessage* handshake = reinterpret_cast<HandshakeMessage*>(message.get());
					std::cout << handshake->GetUniqueID() << std::endl;
					if (handshake->GetUniqueID() == 0)
					{
						unique_id = 0;
						is_init = true;
						std::cout << "Connection is init" << std::endl;
					}
					else
					{
						unique_id = handshake->GetUniqueID();
					}

					break;
				}
			}
		}

		if (unique_id != 0 && !is_init)
			client->SendDirectMessage(HandshakeMessage{ unique_id });
		Sleep(1000);
	}

	client_thread.join();

	return 0;
}
