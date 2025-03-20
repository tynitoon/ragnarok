#include "hasher.hpp"

#include <string>
#include <sstream>
#include <iomanip>

Hasher::Hasher() :
	m_context(EVP_MD_CTX_new())
{
	if (m_context == nullptr)
		throw std::runtime_error("Failed to create the context");
}

Hasher::~Hasher()
{
	EVP_MD_CTX_free(m_context);
}

std::string Hasher::sha256(const std::string& input)
{
	unsigned char hash[EVP_MAX_MD_SIZE];
	unsigned int hash_len;

	if (!EVP_DigestInit_ex(m_context, EVP_sha256(), nullptr) ||
		!EVP_DigestUpdate(m_context, input.c_str(), input.size()) ||
		!EVP_DigestFinal_ex(m_context, hash, &hash_len))
		return "";

	std::stringstream ss;
	for (unsigned int i = 0; i < hash_len; ++i)
		ss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];

	return ss.str();
}