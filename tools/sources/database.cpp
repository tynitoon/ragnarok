#include "database.hpp"

using namespace pqxx;

Database::Database(const std::string &ip) :
	m_connection("dbname=ragnarok user=postgres password=Sdjjsdjj04=t hostaddr=" + ip + " port=5432")
{
	if (!m_connection.is_open())
	{
		std::cerr << "Database::Database: Connection to database failed" << std::endl;
	}
}

bool Database::CheckLogin(std::string_view login, std::string_view password)
{
	std::string query = "SELECT * FROM account WHERE username = '" + std::string(login) + "' AND password = '" + std::string(password) + "'";
	nontransaction non_transaction(m_connection);

	result result;
	try
	{
		result = non_transaction.exec(query);
	}
	catch (const std::exception &e)
	{
		std::cerr << "Database::CheckLogin: " << e.what() << std::endl;
		return false;
	}

	return result.size() == 1;
}