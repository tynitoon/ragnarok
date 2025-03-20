#ifndef HASHER_HPP
#define HASHER_HPP

#include <iostream>
#include <openssl/evp.h>

/*! 
 * \brief Class to hash input
 */
class Hasher
{
public:
	/*!
	 * \brief Initialize the context
	 */
	Hasher();

	/*!
	 * \brief Release the context
	 */
	~Hasher();

	/*!
	 * \brief Compute the SHA256 hash of the input
	 * 
	 * \param[in] input The input to hash
	 * 
	 * \return The SHA256 hash of the input
	 */
	std::string sha256(const std::string& input);

private:
	EVP_MD_CTX* m_context;
};

#endif
