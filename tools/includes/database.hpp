#ifndef DATABASE_HPP
#define DATABASE_HPP

#include <iostream>
#include <string_view>

#include <pqxx/pqxx>

class Database
{
	public:
		/*!
		 * \brief Simple position data
		 */
		struct Point
		{
			int x = 0;	/* X coord */
			int y = 0;	/* Y coord */
		};

		struct Character
		{
			int key = -1;		/* Primary key */
			int account = -1;	/* Foreign key linked to account table */
			int server = -1;	/* Foreign key linked to server table */
			int map = -1;		/* Foreign key linked to map table */
			Point position;		/* Position of the character */
			std::string name;	/* Name of the character */
		};

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
		 * \return Return the key of the account if the login and password are correct, -1 otherwise
		 */
		int CheckLogin(std::string_view login, std::string_view password);

		/*!
		 * \brief Get characters linked to the account
		 *
		 * \param[in] key The key of the account
		 *
		 * \return Return a vector of character
		 */
		std::vector<Character> CharactersFromAccount(int key);

	private:
		pqxx::connection m_connection; /* Connection object linked to the database */
};

#endif
