#include "server.hpp"
//#include <thread>
//#include <iostream>

#include <iostream>

Server::Server(uint32_t port) :
	port_(port),
	acceptor_(io_context_, boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), port)),
	udp_socket_(io_context_, boost::asio::ip::udp::endpoint(boost::asio::ip::udp::v4(), port + 1))
{
	StartAccept();
}

void Server::Run()
{
	io_context_.run();
}

void Server::SendMessage(uint32_t fd, Message&& to_send)
{
	auto buffer = boost::asio::buffer(&to_send, to_send.size);
	std::lock_guard<std::mutex> lock(fd_to_clients_mutex_);
	fd_to_clients_[fd]->socket->async_send(buffer);
}

void Server::SendMessage(std::vector<uint32_t> fds, Message&& to_send)
{
	auto buffer = boost::asio::buffer(&to_send, to_send.size);
	std::lock_guard<std::mutex> lock(fd_to_clients_mutex_);
	for (auto fd : fds)
		fd_to_clients_[fd]->socket->async_send(buffer);
}

deleted_unique_ptr<ReceivedMessage> Server::ReadMessage()
{
	std::lock_guard<std::mutex> lock(message_received_mutex_);
	if (!message_received_queue_.empty())
	{
		deleted_unique_ptr<ReceivedMessage> received_message = std::move(message_received_queue_.front());
		message_received_queue_.pop();
		return received_message;
	}

	return deleted_unique_ptr<ReceivedMessage>(nullptr);
}

void Server::StartAccept()
{
	/* Prepare a new client socket */
	auto socket = std::make_shared<boost::asio::ip::tcp::socket>(io_context_);
	acceptor_.async_accept(*socket,
		[this, socket](boost::system::error_code error)
		{
			if (!error)
			{
				/* Create a new client and add it*/
				auto new_client = std::make_shared<Client>();
				new_client->socket = socket;
				std::string ip = socket->remote_endpoint().address().to_string();
				{
					std::lock_guard<std::mutex> lock(fd_to_clients_mutex_);

					if (ip_to_nbs_.contains(ip))
						++ip_to_nbs_[ip];
					else
						ip_to_nbs_[ip] = port_ + 1; //TODO pas bon car lorsqu'un client se déco ça ne pourra pas libéré le port
					//TODO send connect message to ask client to open a specific port in UDP
					//TODO see if he will have to conf its router or not

					new_client->endpoint = boost::asio::ip::udp::endpoint(socket->remote_endpoint().address(), ip_to_nbs_[ip]);
					fd_to_clients_.insert(std::make_pair(new_client->socket->native_handle(), new_client));
				}

				/* Listen incoming messages from this client */
				HandleClient(new_client);

				/* Prepare a new accept for the next potential connection */
				StartAccept();
			}
			else
				std::cerr << "Error while async_accept: " << error.message() << std::endl;
		});
}

void Server::HandleClient(std::shared_ptr<Client> client)
{
	/* Prepare a buffer to fill */
	auto buffer = boost::asio::buffer(&client->buffer[client->nb_bytes], client->buffer.max_size() - client->nb_bytes);
	client->socket->async_read_some(buffer,
		[this, client](boost::system::error_code error, std::size_t bytes_transferred)
		{
			if (!error)
			{
				client->nb_bytes += bytes_transferred;

				/* Check that we have enough data */
				std::size_t offset = 0;
				while (client->nb_bytes - offset >= sizeof(Message))
				{
					Message* incoming = reinterpret_cast<Message*>(&client->buffer[offset]);
					if (incoming->size > client->buffer.max_size()) /* Error because the message cannot be bigger than the buffer */
					{
						std::cerr << "Drop packet: incoming size = " << incoming->size << std::endl;
						client->nb_bytes = 0;
					}
					else if (client->nb_bytes - offset >= incoming->size) /* The incoming message is complete */
					{
						/* Create a new message and copy incoming data inside */
						std::size_t allocate_size = incoming->size - sizeof(Message) + sizeof(ReceivedMessage);
						deleted_unique_ptr<ReceivedMessage> received_message(reinterpret_cast<ReceivedMessage*>(::operator new(allocate_size)),
							[allocate_size](ReceivedMessage* ptr)
							{
								::operator delete(ptr, allocate_size);
							});
						received_message->fd = client->socket->native_handle();
						memmove(&received_message->message, incoming, incoming->size);

						/* Push our new message in the queue */
						{
							std::lock_guard<std::mutex> lock(message_received_mutex_);
							message_received_queue_.push(std::move(received_message));
						}

						/* This message is handled, we go to the next message */
						offset += incoming->size;
					}
					else
						break;
				}

				/* No more message to handle, wait to receive more bytes from this client */
				HandleClient(client);
			}
			else
			{
				std::cerr << "Connexion closed: " << error.message() << std::endl;
				{
					std::lock_guard<std::mutex> lock(fd_to_clients_mutex_);
					fd_to_clients_.erase(client->socket->native_handle());
				}
			}
		});
}
