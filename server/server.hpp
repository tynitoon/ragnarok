#ifndef SERVER_HPP
#define SERVER_HPP

#include <queue>

#include <boost/asio.hpp>
#include <boost/unordered/unordered_flat_map.hpp>

#include "message.hpp"

template<typename T>
using deleted_unique_ptr = std::unique_ptr<T, std::function<void(T*)>>;

/**
 * @brief Message that contains also the file descriptor of the client
 */
struct ReceivedMessage
{
	uint32_t fd;
	deleted_unique_ptr<Message> message;
};

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

	/**
	 * @brief Send message to a single client
	 * @param fd The file descriptor of the client
	 */
	void SendDirectMessage(uint32_t fd, Message&& to_send);

	/**
	 * @brief Send message to multiple clients
	 * @param fds Vector of file descriptor
	 */
	void SendDirectMessage(std::vector<uint32_t> fds, Message&& to_send);

	deleted_unique_ptr<ReceivedMessage> ReadMessage();

private:
	/**
	 * @brief Client data
	 */
	static constexpr size_t MAX_MESSAGE_SIZE = 4096;	/* Max message size */
	struct Client
	{
		bool is_init = false;
		std::shared_ptr<boost::asio::ip::tcp::socket> socket;	/* Client socket */
		boost::asio::ip::udp::endpoint endpoint;				/* UDP endpoint */
		std::array<char, MAX_MESSAGE_SIZE> buffer;				/* Buffer that contains bytes received from the client by TCP */
		std::size_t nb_bytes = 0;								/* Actual number of data bytes contained in the buffer (from TCP) */
	};

	/**
	 * @brief Asynchronous read UDP handshake
	 * @param buffer Buffer that will be filled with UDP messages
	 */
	void ListenHandshakeUDP(std::array<char, MAX_MESSAGE_SIZE>& buffer);

	/**
	 * @brief Asynchronous accept for new TCP clients
	 */
	void AcceptClient();

	/**
	 * @brief Send Handshake to the client
	 * @param client The TCP client that need an handshake
	 * @param unique_id The ID that we assign to the client (0 to confirm that we received its answer)
	 */
	void SendHandshake(std::shared_ptr<Client> client, uint32_t unique_id);

	/**
	 * @brief Read client messages and store them
	 * @param client Client that is sending messages
	 */
	void HandleClient(std::shared_ptr<Client> client);

	uint32_t m_unique_id;																				/* Used to create a Unique ID per client */
	uint32_t m_sequence_id;																				/* Used for UDP messages */
	boost::asio::io_context m_io_context;																/* I/O boost context */
	boost::asio::ip::tcp::acceptor m_acceptor;															/* Used to accept incoming TCP connexions */
	boost::asio::ip::udp::socket m_udp_socket;															/* Used to handle UDP */
	boost::unordered::unordered_flat_map<uint32_t, std::shared_ptr<Client>> m_id_to_clients;			/* Unique ID to Client object */
	boost::unordered::unordered_flat_map<boost::asio::ip::udp::endpoint, uint32_t> m_id_to_endpoints;	/* Unique ID to UDP endpoint */
	std::queue<std::unique_ptr<ReceivedMessage>> m_message_received_queue;								/* Received messages */
	std::mutex m_id_to_clients_mutex;																	/* Mutex to protect id_to_clients */
	std::mutex m_id_to_endpoints_mutex;																	/* Mutex to protect id_to_endpoints */
	std::mutex m_message_received_mutex;																/* Mutex to protect message queue */
};

#endif