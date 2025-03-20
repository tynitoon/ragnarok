#include "server.hpp"

#include <iostream>

/*!
 * \brief Create a shared_ptr to increase lifetime of to_send data
 *
 * \param[in] msg The message to convert
 *
 * \return A shared_ptr to the message
 */
inline static std::shared_ptr<Message> MessageToSharedPtr(Message&& msg)
{
	uint32_t size = msg.GetSize();
	void* buffer_data = ::operator new(size);
	memmove(buffer_data, &msg, size);
	return std::shared_ptr<Message>(reinterpret_cast<Message*>(buffer_data),
		[size](Message* ptr)
		{
			::operator delete(ptr, size);
		});
}

/*!
 * \brief Callback to handle the send operation
 *
 * \param[in] error The error code
 * \param[in] bytes_transferred The number of bytes transferred
 */
inline static void SendHandler(const boost::system::error_code& error, size_t bytes_transferred)
{
	if (error)
		std::cerr << "Error while send: " << error.message() << std::endl;
	else
		std::cout << "send " << bytes_transferred << " bytes" << std::endl;
}

Server::Server(uint16_t tcp_port, uint16_t udp_port) :
	m_unique_id(1),
	m_acceptor(m_io_context, boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), tcp_port))
{
	AcceptClient();
}

void Server::Run()
{
	m_io_context.run();
}

void Server::SendMessage(uint32_t id, Message&& to_send)
{
	auto msg_ptr = MessageToSharedPtr(std::move(to_send));
	auto buffer = boost::asio::buffer(msg_ptr.get(), msg_ptr->GetSize());

	try
	{
		std::lock_guard<std::mutex> lock(m_id_to_clients_mutex);
		if (m_id_to_clients.contains(id))
		{
			m_id_to_clients[id]->socket->async_send(buffer, [msg_ptr](const boost::system::error_code& error, size_t bytes_transferred)
				{
					SendHandler(error, bytes_transferred);
				});
		}
	}
	catch (const std::exception& e)
	{
		std::cerr << "Server::SendMessage Error: " << e.what() << std::endl;
	}
}

void Server::SendMessage(const std::vector<uint32_t>& ids, Message&& to_send)
{
	auto msg_ptr = MessageToSharedPtr(std::move(to_send));
	auto buffer = boost::asio::buffer(msg_ptr.get(), msg_ptr->GetSize());
	try
	{
		std::lock_guard<std::mutex> lock(m_id_to_clients_mutex);
		for (auto id : ids)
		{
			if (m_id_to_clients.contains(id))
			{
				m_id_to_clients[id]->socket->async_send(buffer, [msg_ptr](const boost::system::error_code& error, size_t bytes_transferred)
					{
						SendHandler(error, bytes_transferred);
					});
			}
		}
	}
	catch (const std::exception& e)
	{
		std::cerr << "Server::SendMessage (vec) Error: " << e.what() << std::endl;
	}
}

std::unique_ptr<MessageFrom> Server::ReadMessage()
{
	try
	{
		std::lock_guard<std::mutex> lock(m_message_received_mutex);
		if (!m_message_received_queue.empty())
		{
			std::unique_ptr<MessageFrom> message = std::move(m_message_received_queue.front());
			m_message_received_queue.pop();

			return message;
		}
	}
	catch (const std::exception& e)
	{
		std::cerr << "Server::ReadMessage Error: " << e.what() << std::endl;
	}

	return std::unique_ptr<MessageFrom>(nullptr);
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
			std::cout << "New client" << std::endl;
			std::shared_ptr<Client> new_client = std::make_shared<Client>();
			new_client->unique_id = m_unique_id++;
			new_client->socket = socket;
			{
				std::lock_guard<std::mutex> lock(m_id_to_clients_mutex);
				m_id_to_clients[new_client->unique_id] = new_client;
			}

			/* Listen incoming messages from this client */
			HandleClient(new_client);

			/* Prepare a new accept for the next potential connection */
			AcceptClient();
		});
}

void Server::HandleClient(const std::shared_ptr<Client>& client)
{
	/* Prepare a buffer to fill */
	auto buffer = boost::asio::buffer(&client->buffer[client->nb_bytes], client->buffer.max_size() - client->nb_bytes);
	client->socket->async_read_some(buffer,
		[this, client](boost::system::error_code error, std::size_t bytes_transferred)
		{
			if (error)
			{
				std::cerr << "Connexion closed: " << error.message() << " ID = " << client->unique_id << std::endl;
				{
					std::lock_guard<std::mutex> lock(m_id_to_clients_mutex);
					m_id_to_clients.erase(client->unique_id);
				}
				return;
			}

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
						m_message_received_queue.push(std::make_unique<MessageFrom>(MessageFrom{ client->unique_id, std::move(new_message) }));
					}

					/* This message is handled, we go to the next message */
					offset += message_size;
				}
				else
					break;
			}
			client->nb_bytes -= offset;
			if (offset < client->buffer.max_size())
				memmove(&client->buffer[0], &client->buffer[offset], client->nb_bytes);

			/* No more message to handle, wait to receive more bytes from this client */
			HandleClient(client);
		});
}
