#ifndef CLIENT_HPP
#define CLIENT_HPP

#include <queue>
#include <thread>

#include <boost/asio.hpp>
#include <boost/unordered/unordered_flat_map.hpp>

#include "message.hpp"

template<typename T>
using deleted_unique_ptr = std::unique_ptr<T, std::function<void(T*)>>;

/*!
 * \brief UDP / TCP Client
 */
class Client
{
public:
	/*!
	 * \brief Client constructor
	 * 
	 * \param[in] ip The ip address of the server
	 * \param[in] tcp_port Port used for TCP on which the server is listening
	 * \param[in] udp_port Port used for UDP on which the server is listening
	 */
	Client(std::string ip, uint16_t tcp_port, uint16_t udp_port);

	/*!
	 * \brief Start event loop of the client
	 */
	void Run();

	/*!
	 * \brief Send message to the server using TCP
	 */
	void SendMessage(Message&& to_send);

	/*!
	 * \brief Send message to the server using UDP
	 */
	void SendDirectMessage(Message&& to_send);

	/*!
	 * \brief Get a message from the receive queue after removing it (nullptr if empty)
	 * 
	 * \return Return a received message
	 */
	deleted_unique_ptr<Message> ReadMessage();

private:
	/*!
	 * \brief Read server messages and store them
	 */
	void HandleReceiveUDP();

	/*!
	 * \brief Read server messages and store them
	 */
	void HandleReceiveTCP();

	/*!
	 * \brief Perform a send with the unique ID to the server every 250ms
	 *
	 * \param[in] unique_id The unique ID to send to the server
	 */
	void HandshakeLoop(uint32_t unique_id) noexcept;

	static constexpr size_t MAX_MESSAGE_SIZE = 4096;					/* Max message size */
	bool m_is_init;														/* True if the client is connected with TCP and UDP */
	std::atomic<bool> m_handshake_is_running;							/* To know if we are trying to handshake with the server */
	std::array<char, MAX_MESSAGE_SIZE> m_tcp_buffer;					/* Buffer that contains bytes received from the server by TCP */
	std::array<char, MAX_MESSAGE_SIZE> m_udp_buffer;					/* Buffer that contains bytes received from the server by UDP */
	std::size_t m_tcp_nb_bytes;											/* Actual number of data bytes contained in the buffer (from TCP) */
	std::size_t m_udp_nb_bytes;											/* Actual number of data bytes contained in the buffer (from UDP) */
	std::uint32_t m_highest_sequence_id ;								/* Highest sequence ID received from server */
	boost::asio::io_context m_io_context;								/* I/O boost context */
	boost::asio::ip::tcp::socket m_tcp_socket;							/* Used to handle TCP*/
	boost::asio::ip::udp::socket m_udp_socket;							/* Used to receive UDP messages */
	boost::asio::ip::udp::endpoint m_server_endpoint;					/* Used to send UDP messages */
	boost::asio::ip::udp::endpoint m_remote_endpoint;					/* Endpoint that is filled when we receive UDP messages */
	std::queue<deleted_unique_ptr<Message>> m_message_received_queue;	/* Received messages */
	std::mutex m_socket_mutex;											/* Mutex to protect message queue */
	std::mutex m_message_received_mutex;								/* Mutex to protect message queue */

};

#endif