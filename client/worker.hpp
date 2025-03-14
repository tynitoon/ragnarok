#ifndef WORKER_HPP
#define WORKER_HPP

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

		const std::shared_ptr<Client> &m_client;	/* Client object that handle the connection with the server */
};

#endif
