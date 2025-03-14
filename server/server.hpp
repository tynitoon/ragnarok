#ifndef SERVER_HPP
#define SERVER_HPP

#include <queue>

#include <boost/asio.hpp>
#include <boost/unordered/unordered_flat_map.hpp>

#include "message.hpp"

template<typename T>
using deleted_unique_ptr = std::unique_ptr<T, std::function<void(T*)>>;

/*!
 * \brief Message that contains also the file descriptor of the client
 */
struct MessageFrom
{
	uint32_t fd;
	deleted_unique_ptr<Message> message;
};

/*!
 * \brief UDP / TCP Server
 */
class Server
{
public:
	/*!
	 * \brief TcpServer constructor
	 * 
	 * \param[in] tcp_port Port number used for TCP on which the server is listening
	 * \param[in] udp_port Port number used for UDP on which the server is listening
	 */
	Server(uint16_t tcp_port, uint16_t udp_port);

	/*!
	 * \brief Run the event loop of the server
	 */
	void Run();

	/*!
	 * \brief Send message to a single client
	 * 
	 * \param[in] fd The file descriptor of the client
	 */
	void SendMessage(uint32_t fd, Message&& to_send);

	/*!
	 * \brief Send message to multiple clients
	 * 
	 * \param[in] fds Vector of file descriptor
	 */
	void SendMessage(const std::vector<uint32_t> &fds, Message&& to_send);

	/*!
	 * \brief Send message to a single client
	 * 
	 * \param[in] fd The file descriptor of the client
	 */
	void SendDirectMessage(uint32_t fd, Message&& to_send);

	/*!
	 * \brief Send message to multiple clients
	 * 
	 * \param[in] fds Vector of file descriptor
	 */
	void SendDirectMessage(const std::vector<uint32_t> &fds, Message&& to_send);

	/*!
	 * \brief Get a message from the receive queue after removing it (nullptr if empty)
	 * 
	 * \return Return a received message
	 */
	std::unique_ptr<MessageFrom> ReadMessage();

private:
	/*!
	 * \brief Client data
	 */
	static constexpr size_t MAX_MESSAGE_SIZE = 4096;	/* Max message size */
	struct Client
	{
		bool is_init = false;									/* True if the client is connected in TCP and UDP */
		uint32_t unique_id = 0;									/* Unique ID of the client */
		std::shared_ptr<boost::asio::ip::tcp::socket> socket;	/* Client socket */
		boost::asio::ip::udp::endpoint endpoint;				/* UDP endpoint */
		std::array<char, MAX_MESSAGE_SIZE> buffer;				/* Buffer that contains bytes received from the client by TCP */
		std::size_t nb_bytes = 0;								/* Actual number of data bytes contained in the buffer (from TCP) */
	};

	/*!
	 * \brief Asynchronous read UDP handshake
	 *
	 * \param[in] buffer Buffer that will be filled with UDP messages
	 */
	void ListenHandshakeUDP();

	/*!
	 * \brief Asynchronous accept for new TCP clients
	 */
	void AcceptClient();

	/*!
	 * \brief Read client messages and store them
	 * 
	 * \param[in] client Client that is sending messages
	 */
	void HandleClient(const std::shared_ptr<Client>& client);

	uint32_t m_unique_id;																				/* Used to create a Unique ID per client */
	uint32_t m_sequence_id;																				/* Used for UDP messages */
	boost::asio::io_context m_io_context;																/* I/O boost context */
	boost::asio::ip::tcp::acceptor m_acceptor;															/* Used to accept incoming TCP connexions */
	boost::asio::ip::udp::socket m_udp_socket;															/* Used to handle UDP */
	boost::asio::ip::udp::endpoint m_remote_endpoint;													/* Endpoint that is filled when we receive UDP messages */
	std::array<char, MAX_MESSAGE_SIZE> m_udp_buffer;													/* Buffer that contains bytes received from clients by UDP (Non circular, we only need it for handshake) */
	boost::unordered::unordered_flat_map<uint32_t, std::shared_ptr<Client>> m_id_to_clients;			/* Unique ID to Client object */
	boost::unordered::unordered_flat_map<boost::asio::ip::udp::endpoint, uint32_t> m_id_to_endpoints;	/* Unique ID to UDP endpoint */
	std::queue<std::unique_ptr<MessageFrom>> m_message_received_queue;									/* Received messages */
	std::mutex m_id_to_clients_mutex;																	/* Mutex to protect id_to_clients */
	std::mutex m_id_to_endpoints_mutex;																	/* Mutex to protect id_to_endpoints */
	std::mutex m_message_received_mutex;																/* Mutex to protect message queue */


};

#endif