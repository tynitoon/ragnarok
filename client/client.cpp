#include "client.hpp"

#include <iostream>

Client::Client(std::string ip, uint32_t port) :
	m_tcp_socket(m_io_context),
	m_udp_socket(m_io_context, boost::asio::ip::udp::endpoint(boost::asio::ip::udp::v4(), 0)),
	m_server_endpoint(boost::asio::ip::address::from_string(ip), port + 1)
{
	m_tcp_socket.connect(boost::asio::ip::tcp::endpoint(boost::asio::ip::address::from_string(ip), port));
	HandleReceiveTCP();
	HandleReceiveUDP();
}

void Client::Run()
{
	m_io_context.run();
}

void Client::SendMessage(Message&& to_send)
{
	auto buffer = boost::asio::buffer(&to_send, to_send.GetSize());
	std::lock_guard<std::mutex> lock(m_socket_mutex);
	m_tcp_socket.async_send(buffer, [](const boost::system::error_code& error, size_t bytes_transferred) {});
}

void Client::SendDirectMessage(Message&& to_send)
{
	auto buffer = boost::asio::buffer(&to_send, to_send.GetSize());
	std::lock_guard<std::mutex> lock(m_socket_mutex);
	m_udp_socket.async_send_to(buffer, m_server_endpoint, [](const boost::system::error_code& error, size_t bytes_transferred) { std::cout << "send " << bytes_transferred << std::endl; });
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
	boost::asio::ip::udp::endpoint remote_endpoint;
	/* Prepare a buffer to fill */
	auto buffer = boost::asio::buffer(&m_udp_buffer[m_udp_nb_bytes], m_udp_buffer.max_size() - m_udp_nb_bytes);
	m_udp_socket.async_receive_from(boost::asio::buffer(buffer), remote_endpoint,
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
				std::cerr << "Connexion closed: " << error.message() << std::endl;
				return;
			}

			std::cout << "received bytes = " << bytes_transferred << std::endl;
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
				}
				else if (m_tcp_nb_bytes - offset >= message_size) /* The incoming message is complete */
				{
					/* Create a new message and copy incoming data inside */
					deleted_unique_ptr<Message> new_message(reinterpret_cast<Message*>(::operator new(message_size)),
						[message_size](Message* ptr)
						{
							::operator delete(ptr, message_size);
						});
					memmove(new_message.get(), incoming, message_size);

					std::cout << "full message " << new_message->GetSequenceID() << " " << new_message->GetSize() << " " << static_cast<uint32_t>(new_message->GetType()) << std::endl;
					/* Push our new message in the queue */
					{
						std::lock_guard<std::mutex> lock(m_message_received_mutex);
						m_message_received_queue.push(std::move(new_message));
					}

					/* This message is handled, we go to the next message */
					offset += message_size;
				}
				else
					break;
			}
			m_tcp_nb_bytes -= offset;
			memmove(&m_tcp_buffer[0], &m_tcp_buffer[offset], m_tcp_nb_bytes);

			/* No more message to handle, wait to receive more bytes from server */
			HandleReceiveTCP();
		});
}
