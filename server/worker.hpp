#pragma once
#ifndef WORKER_HPP
#define WORKER_HPP

#include "server.hpp"

class Worker
{
public:
	/*!
	 * \brief Worker constructor
	 * 
	 * \param[in] server The server object that is communicating with all clients
	 */
	Worker(const std::shared_ptr<Server>& server) noexcept;

	/*!
	 * @brief Handle messages from clients
	 */
	void Run();

private:
	const std::shared_ptr<Server>& m_server; /* Server object that handle connection with all clients */
};

#endif
