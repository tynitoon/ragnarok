#include "server.hpp"

#include <iostream>

Server::Server(uint32_t port) :
	m_unique_id(0),
	m_sequence_id(0),
	m_acceptor(m_io_context, boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), port)),
	m_udp_socket(m_io_context, boost::asio::ip::udp::endpoint(boost::asio::ip::udp::v4(), port + 1))
{
	std::array<char, MAX_MESSAGE_SIZE> buffer;
	ListenHandshakeUDP(buffer);
	AcceptClient();
}

void Server::Run()
{
	m_io_context.run();
}

void Server::SendMessage(uint32_t id, Message&& to_send)
{
	auto buffer = boost::asio::buffer(&to_send, to_send.GetSize());
	std::lock_guard<std::mutex> lock(m_id_to_clients_mutex);
	for (auto kv : m_id_to_clients)
		kv.second->socket->async_send(buffer, [](const boost::system::error_code& error, size_t bytes_transferred) {});
	//if (m_id_to_clients.contains(id))
	//	m_id_to_clients[id]->socket->async_send(buffer, [](const boost::system::error_code& error, size_t bytes_transferred) {});
}

void Server::SendMessage(std::vector<uint32_t> ids, Message&& to_send)
{
	auto buffer = boost::asio::buffer(&to_send, to_send.GetSize());
	std::lock_guard<std::mutex> lock(m_id_to_clients_mutex);
	for (auto id : ids)
	{
		if (m_id_to_clients.contains(id))
			m_id_to_clients[id]->socket->async_send(buffer, [](const boost::system::error_code& error, size_t bytes_transferred) {});
	}
}

void Server::SendDirectMessage(uint32_t id, Message&& to_send)
{
	auto buffer = boost::asio::buffer(&to_send, to_send.GetSize());
	std::lock_guard<std::mutex> lock(m_id_to_clients_mutex);
	if (m_id_to_clients.contains(id))
	{
		to_send.SetSequenceID(m_sequence_id++);
		m_udp_socket.async_send_to(buffer, m_id_to_clients[id]->endpoint, [](const boost::system::error_code& error, size_t bytes_transferred) {});
	}
}

void Server::SendDirectMessage(std::vector<uint32_t> ids, Message&& to_send)
{
	auto buffer = boost::asio::buffer(&to_send, to_send.GetSize());
	std::lock_guard<std::mutex> lock(m_id_to_clients_mutex);
	to_send.SetSequenceID(m_sequence_id++);
	for (auto id : ids)
	{
		if (m_id_to_clients.contains(id))
			m_udp_socket.async_send_to(buffer, m_id_to_clients[id]->endpoint, [](const boost::system::error_code& error, size_t bytes_transferred) {});
	}
}

deleted_unique_ptr<ReceivedMessage> Server::ReadMessage()
{
	std::lock_guard<std::mutex> lock(m_message_received_mutex);
	if (!m_message_received_queue.empty())
	{
		deleted_unique_ptr<ReceivedMessage> received_message = std::move(m_message_received_queue.front());
		m_message_received_queue.pop();
		return received_message;
	}

	return deleted_unique_ptr<ReceivedMessage>(nullptr);
}

void Server::ListenHandshakeUDP(std::array<char, MAX_MESSAGE_SIZE>& buffer)
{
	boost::asio::ip::udp::endpoint remote_endpoint;
	m_udp_socket.async_receive_from(
		boost::asio::buffer(buffer), remote_endpoint,
		[&](const boost::system::error_code& error, std::size_t bytes_transferred)
		{
			if (error)
			{
				std::cerr << "Error while UDP async receive: " << error.message() << std::endl;
				return;
			}

			/* Check that we have enough data for Handshake */
			if (bytes_transferred == sizeof(HandshakeMessage))
			{
				HandshakeMessage* incoming = reinterpret_cast<HandshakeMessage*>(buffer.data());
				if (incoming->GetType() == MessageType::HANDSHAKE && incoming->GetSize() == sizeof(HandshakeMessage))
				{
					std::lock_guard<std::mutex> lock(m_id_to_clients_mutex);
					if (m_id_to_clients.contains(incoming->GetUniqueID()))
					{
						std::shared_ptr<Client> client = m_id_to_clients[incoming->GetUniqueID()];
						client->is_init = true;
						client->endpoint = remote_endpoint;
						SendHandshake(client, 0);
					}
				}
			}

			ListenHandshakeUDP(buffer);
		});
}

void Server::AcceptClient()
{
	/* Prepare a new client socket */
	auto socket = std::make_shared<boost::asio::ip::tcp::socket>(m_io_context);
	m_acceptor.async_accept(*socket,
		[this, socket](boost::system::error_code error)
		{
			if (error)
			{
				std::cerr << "Error while async_accept: " << error.message() << std::endl;
				return;
			}

			/* Create a new client and add it*/
			std::shared_ptr<Client> new_client = std::make_shared<Client>();
			new_client->socket = socket;
			{
				std::lock_guard<std::mutex> lock(m_id_to_clients_mutex);
				m_id_to_clients[static_cast<uint32_t>(socket->native_handle())] = new_client;
			}

			/* Send the Handshake to the client*/
			std::cout << "send handshake" << std::endl;
			SendHandshake(new_client, static_cast<uint32_t>(socket->native_handle()));

			/* Listen incoming messages from this client */
			HandleClient(new_client);

			/* Prepare a new accept for the next potential connection */
			AcceptClient();
		});
}

void Server::SendHandshake(std::shared_ptr<Client> client, uint32_t unique_id)
{
	HandshakeMessage message(unique_id);
	auto buffer = boost::asio::buffer(&message, message.GetSize());
	client->socket->async_send(buffer, [](const boost::system::error_code& error, size_t bytes_transferred) {});
}

void Server::HandleClient(std::shared_ptr<Client> client)
{
	/* Prepare a buffer to fill */
	auto buffer = boost::asio::buffer(&client->buffer[client->nb_bytes], client->buffer.max_size() - client->nb_bytes);
	client->socket->async_read_some(buffer,
		[this, client](boost::system::error_code error, std::size_t bytes_transferred)
		{
			if (error)
			{
				std::cerr << "Connexion closed: " << error.message() << std::endl;
				{
					std::lock_guard<std::mutex> lock(m_id_to_clients_mutex);
					m_id_to_clients.erase(static_cast<uint32_t>(client->socket->native_handle()));
				}
				return;
			}

			if (client->is_init)
			{
				client->nb_bytes += bytes_transferred;

				/* Check that we have enough data */
				std::size_t offset = 0;
				while (client->nb_bytes - offset >= sizeof(Message))
				{
					Message* incoming = reinterpret_cast<Message*>(&client->buffer[offset]);
					uint32_t message_size = incoming->GetSize();
					if (message_size > client->buffer.max_size()) /* Error because the message cannot be bigger than the buffer */
					{
						std::cerr << "Drop packet: incoming size = " << message_size << std::endl;
						client->nb_bytes = 0;
					}
					else if (client->nb_bytes - offset >= message_size) /* The incoming message is complete */
					{
						/* Create a new message and copy incoming data inside */
						deleted_unique_ptr<Message> new_message(reinterpret_cast<Message*>(::operator new(message_size)),
							[message_size](Message* ptr)
							{
								::operator delete(ptr, message_size);
							});
						memmove(new_message.get(), incoming, message_size);

						/* Push our new message in the queue */
						{
							std::lock_guard<std::mutex> lock(m_message_received_mutex);
							m_message_received_queue.push(std::make_unique<ReceivedMessage>(ReceivedMessage{ static_cast<uint32_t>(client->socket->native_handle()), std::move(new_message) }));
						}

						/* This message is handled, we go to the next message */
						offset += message_size;
					}
					else
						break;
				}
				client->nb_bytes -= offset;
				memmove(&client->buffer[0], &client->buffer[offset], client->nb_bytes);
			}

			/* No more message to handle, wait to receive more bytes from this client */
			HandleClient(client);
		});
}
