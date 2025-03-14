#include "client.hpp"

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

Client::Client(std::string ip, uint16_t tcp_port, uint16_t udp_port) :
	m_is_init(false),
	m_handshake_is_running(false),
	m_tcp_buffer(),
	m_udp_buffer(),
	m_tcp_nb_bytes(0),
	m_udp_nb_bytes(0),
	m_highest_sequence_id(0),
	m_tcp_socket(m_io_context),
	m_udp_socket(m_io_context, boost::asio::ip::udp::endpoint(boost::asio::ip::udp::v4(), 0)),
	m_server_endpoint(boost::asio::ip::address::from_string(ip), udp_port)
{
	std::cout << "Trying to connect to " << ip << ", TCP port:" << tcp_port << " UDP port:" << udp_port << std::endl;
	m_tcp_socket.async_connect(boost::asio::ip::tcp::endpoint(boost::asio::ip::address::from_string(ip), tcp_port),
		[this](boost::system::error_code error)
		{
			if (error)
			{
				std::cerr << "Error while async_connect: " << error.message() << std::endl;
				//TODO : Send a message to workers that ask to write a popup that contains a button do quit the client
				return;
			}

			/* Start TCP and UDP handles if we are connected */
			HandleReceiveTCP();
			HandleReceiveUDP();
		});
}

void Client::Run()
{
	m_io_context.run();
}

void Client::SendMessage(Message&& to_send)
{
	auto msg_ptr = MessageToSharedPtr(std::move(to_send));
	auto buffer = boost::asio::buffer(msg_ptr.get(), msg_ptr->GetSize());
	std::lock_guard<std::mutex> lock(m_socket_mutex);
	m_tcp_socket.async_send(buffer, [msg_ptr](const boost::system::error_code& error, size_t bytes_transferred)
		{
			SendHandler(error, bytes_transferred);
		});
}

void Client::SendDirectMessage(Message&& to_send)
{
	auto msg_ptr = MessageToSharedPtr(std::move(to_send));
	auto buffer = boost::asio::buffer(msg_ptr.get(), msg_ptr->GetSize());
	std::lock_guard<std::mutex> lock(m_socket_mutex);
	m_udp_socket.async_send_to(buffer, m_server_endpoint, [msg_ptr](const boost::system::error_code& error, size_t bytes_transferred)
		{
			SendHandler(error, bytes_transferred);
		});
}

deleted_unique_ptr<Message> Client::ReadMessage()
{
	std::lock_guard<std::mutex> lock(m_message_received_mutex);
	if (!m_message_received_queue.empty())
	{
		deleted_unique_ptr<Message> received_message = std::move(m_message_received_queue.front());
		m_message_received_queue.pop();
		return received_message;
	}

	return deleted_unique_ptr<Message>(nullptr);
}

void Client::HandleReceiveUDP()
{
	/* Prepare a buffer to fill */
	auto buffer = boost::asio::buffer(&m_udp_buffer[m_udp_nb_bytes], m_udp_buffer.max_size() - m_udp_nb_bytes);
	m_udp_socket.async_receive_from(boost::asio::buffer(buffer), m_remote_endpoint,
		[this](const boost::system::error_code& error, size_t bytes_transferred)
		{
			if (error)
			{
				std::cerr << "Error while UDP async receive: " << error.message() << std::endl;
				return;
			}

			m_udp_nb_bytes += bytes_transferred;

			/* Check that we have enough data */
			std::size_t offset = 0;
			while (m_udp_nb_bytes - offset >= sizeof(Message))
			{
				Message* incoming = reinterpret_cast<Message*>(&m_udp_buffer[offset]);
				uint32_t message_size = incoming->GetSize();
				if (message_size > m_udp_buffer.max_size()) /* Error because the message cannot be bigger than the buffer */
				{
					std::cerr << "Drop packet: incoming size = " << message_size << std::endl;
					m_udp_nb_bytes = 0;
				}
				else if (m_udp_nb_bytes - offset >= message_size) /* The incoming message is complete */
				{
					if (incoming->GetSequenceID() > m_highest_sequence_id)
					{
						m_highest_sequence_id = incoming->GetSequenceID();
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
							m_message_received_queue.push(std::move(new_message));
						}
					}

					/* This message is handled, we go to the next message */
					offset += message_size;
				}
				else
					break;
			}
			m_udp_nb_bytes -= offset;
			if (offset < m_udp_buffer.max_size())
				memmove(&m_udp_buffer[0], &m_udp_buffer[offset], m_udp_nb_bytes);

			HandleReceiveUDP();
		});
}

void Client::HandleReceiveTCP()
{
	/* Prepare a buffer to fill */
	auto buffer = boost::asio::buffer(&m_tcp_buffer[m_tcp_nb_bytes], m_tcp_buffer.max_size() - m_tcp_nb_bytes);
	m_tcp_socket.async_read_some(buffer,
		[this](boost::system::error_code error, std::size_t bytes_transferred)
		{
			if (error)
			{
				std::cerr << "Connection closed: " << error.message() << std::endl;
				return;
			}

			m_tcp_nb_bytes += bytes_transferred;

			/* Check that we have enough data */
			std::size_t offset = 0;
			while (m_tcp_nb_bytes - offset >= sizeof(Message))
			{
				Message* incoming = reinterpret_cast<Message*>(&m_tcp_buffer[offset]);
				uint32_t message_size = incoming->GetSize();
				if (message_size > m_tcp_buffer.max_size()) /* Error because the message cannot be bigger than the buffer */
				{
					std::cerr << "Drop packet: incoming size = " << message_size << std::endl;
					m_tcp_nb_bytes = 0;
					offset = 0;
				}
				else if (m_tcp_nb_bytes - offset >= message_size) /* The incoming message is complete */
				{
					if (m_is_init)
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
							m_message_received_queue.push(std::move(new_message));
						}
					}
					else if (incoming->GetType() == MessageType::HANDSHAKE)
					{
						HandshakeMessage* handshake = reinterpret_cast<HandshakeMessage*>(incoming);
						std::cout << handshake->GetUniqueID() << std::endl;
						if (handshake->GetUniqueID() == 0)
						{
							m_handshake_is_running = false;
							m_is_init = true;
							std::cout << "Connection is initialized" << std::endl;
						}
						else if (!m_handshake_is_running)
						{
							std::thread handshake_thread(&Client::HandshakeLoop, this, handshake->GetUniqueID());
							handshake_thread.detach();
							m_handshake_is_running = true;
						}
					}

					/* This message is handled, we go to the next message */
					offset += message_size;
				}
				else
					break;
			}
			m_tcp_nb_bytes -= offset;
			if (offset < m_tcp_buffer.max_size())
				memmove(&m_tcp_buffer[0], &m_tcp_buffer[offset], m_tcp_nb_bytes);

			/* No more message to handle, wait to receive more bytes from server */
			HandleReceiveTCP();
		});
}

void Client::HandshakeLoop(uint32_t unique_id) noexcept
{
	while (m_handshake_is_running)
	{
		SendDirectMessage(HandshakeMessage{ unique_id });
		std::this_thread::sleep_for(std::chrono::milliseconds(250));
	}
}