#ifndef SERVER_HPP
#define SERVER_HPP


#include <queue>

#include <boost/asio.hpp>
#include <boost/unordered/unordered_flat_map.hpp>

#include "message.hpp"

/**
 * @brief Client data
 */
struct Client
{
	std::shared_ptr<boost::asio::ip::tcp::socket> socket;	/* Client socket */
	boost::asio::ip::udp::endpoint endpoint;				/* Endpoint to send UDP message to this client */
	std::array<char, 4096> buffer;							/* Buffer that contains bytes received from the client */
	std::size_t nb_bytes;									/* Number of bytes received but not handled present in the buffer */
};

/**
 * @brief Message that contains also the file descriptor of the client
 */
__pragma(pack(push, 1))
struct ReceivedMessage
{
	uint32_t fd;
	Message message;
};
__pragma(pack(pop))

template<typename T>
using deleted_unique_ptr = std::unique_ptr<T, std::function<void(T*)>>;

/**
 * @brief UDP / TCP Server
 */
class Server
{
public:
	/**
	 * @brief Constructeur de la classe TcpServer.
	 * @param port Le numéro de port sur lequel le serveur écoute.
	 */
	Server(uint32_t port);

	/**
	 * @brief Méthode pour lancer la boucle d'événements du serveur.
	 */
	void Run();

	/**
	 * @brief Send message to a single client
	 * @param fd The file descriptor of the client
	 */
	void SendMessage(uint32_t fd, Message&& to_send);

	/**
	 * @brief Send message to multiple clients
	 * @param fds Vector of file descriptor
	 */
	void SendMessage(std::vector<uint32_t> fds, Message&& to_send);

	deleted_unique_ptr<ReceivedMessage> ReadMessage();

private:
	/**
	 * @brief Asynchronous accept for new clients
	 */
	void StartAccept();

	/**
	 * @brief Read client messages and store them
	 * @param client Client that is sending messages
	 */
	void HandleClient(std::shared_ptr<Client> client);

	static constexpr size_t max_length = 4096;												/* Max message size */
	uint32_t port_;																			/* Port used to listen TCP, (port + 1 for UDP) */
	boost::asio::io_context io_context_;													/* I/O boost context */
	boost::asio::ip::tcp::acceptor acceptor_;												/* Used to accept incoming TCP connexions */
	boost::asio::ip::udp::socket udp_socket_;												/* Used to handle UDP */
	boost::unordered::unordered_flat_map<uint32_t, std::shared_ptr<Client>> fd_to_clients_;	/* File descriptor to Client object */
	boost::unordered::unordered_flat_map<std::string, uint32_t> ip_to_nbs_;					/* Count the number of the same IP (used to compute UDP port) */
	std::queue<deleted_unique_ptr<ReceivedMessage>> message_received_queue_;				/* Received messages */
	std::mutex fd_to_clients_mutex_;														/* Mutex to protect fd_to_clients */
	std::mutex message_received_mutex_;														/* Mutex to protect message queue */
};

#endif