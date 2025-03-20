#ifndef DATABASE_HPP
#define DATABASE_HPP

#include <iostream>
#include <string_view>

#include <pqxx/pqxx>

class Database
{
	public:
		/*!
		 * \brief Database constructor
		 *
		 * \param[in] ip The IP address of the database
		 */
		Database(const std::string& ip);

		/*!
		 * \brief Check if the login and password are correct
		 *
		 * \param[in] login The login of the user
		 * \param[in] password The password of the user
		 *
		 * \return Return true if the login and password are correct
		 */
		bool CheckLogin(std::string_view login, std::string_view password);

	private:
		pqxx::connection m_connection;
};

#endif
