#ifndef WORKER_HPP
#define WORKER_HPP

#include "database.hpp"
#include "hasher.hpp"
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
	 * \brief Handle messages from clients
	 */
	void Run();

private:
	/*!
	 * \brief Handle login message
	 * 
	 * \param[in] fd The file descriptor of the client
	 * \param[in] login The login message
	 */
	void HandleLogin(uint32_t fd, const LoginMessage& login);

	const std::shared_ptr<Server>& m_server;	/* Server object that handle connection with all clients */
	Database m_database;						/* Database object that handle connection with the database */
	Hasher m_hasher;							/* Hasher object that is used to generate hash */
};

#endif
