#ifndef WORKER_HPP
#define WORKER_HPP

#include "client.hpp"

class Worker
{
	public:
		/**
		 * @brief Worker constructor
		 * @param client The client object that is communicating with the server
		 */
		Worker(const std::shared_ptr<Client> &client) noexcept;

		/**
		 * @brief Handle messages from the server
		 */
		void Run();

	private:
		/**
		 * @brief Perform the handshake with the server so it can link our TCP and UDP connexions to the same object
		 * @param message Contains the handshake information
		 */
		void Handshake(HandshakeMessage& message);

		/**
		 * @brief Perform a send with the unique ID to the server every 250ms
		 * @param unique_id The unique ID to send to the server
		 */
		void HandshakeLoop(uint32_t unique_id) noexcept;

		const std::shared_ptr<Client> &m_client; /* Client object that handle the connection with the server */
		std::atomic<bool> m_handshake_is_running; /* To know if we are trying to handshake with the server */
};

#endif
