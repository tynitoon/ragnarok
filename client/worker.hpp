#ifndef WORKER_HPP
#define WORKER_HPP

#include <atomic>

#include "client.hpp"

class Worker
{
	public:
		/*!
		 * \brief Worker constructor
		 * 
		 * \param[in] client The client object that is communicating with the server
		 */
		Worker(const std::shared_ptr<Client> &client) noexcept;

		/*!
		 * \brief Handle messages from the server
		 */
		void Run();

	private:
		/*!
		 * \brief Handle the handshake message from the server
		 *
		 * \param[in] handshake The handshake message received from the server
		 */
		void HandleHandshake(const HandshakeMessage& handshake);

		/*!
		 * \brief Perform a send with the unique ID to the server every 250ms
		 *
		 * \param[in] unique_id The unique ID to send to the server
		 */
		void HandshakeLoop(uint32_t unique_id) noexcept;

		/*!
		 * \brief Handle an error message from the server
		 *
		 * \param[in] error The error message received from the server
		 */
		void HandleError(const ErrorMessage& error);

		const std::shared_ptr<Client> &m_client;	/* Client object that handle the connection with the server */
		std::atomic<bool> m_handshake_is_running;	/* To know if we are trying to handshake with the server */
};

#endif
