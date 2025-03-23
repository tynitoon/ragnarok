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

int Database::CheckLogin(std::string_view login, std::string_view password)
{
	std::string query = "SELECT key FROM account WHERE username = '" + std::string(login) + "' AND password = '" + std::string(password) + "'";
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

	if (result.size() == 1)
		return result[0][0].as<int>();

	return -1;
}

std::vector<Database::Character> Database::CharactersFromAccount(int key)
{
	std::string query = "SELECT * FROM character WHERE account = " + std::to_string(key);
	nontransaction non_transaction(m_connection);

	result result;
	try
	{
		result = non_transaction.exec(query);
	}
	catch (const std::exception& e)
	{
		std::cerr << "Database::CheckLogin: " << e.what() << std::endl;
		return {};
	}

	std::vector<Database::Character> characters;
	for (std::size_t row = 0; row < std::size(result); ++row)
		characters.push_back(Database::Character{ result[row][0].as<int>(), result[row][1].as<int>(), result[row][2].as<int>(), result[row][3].as<int>(), result[row][4].as<int>(), result[row][5].c_str()});
	//FIX ME and handle point for position

	return characters;
}