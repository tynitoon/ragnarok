#ifndef MESSAGE_HPP
#define MESSAGE_HPP

#undef ERROR	/* Avoid conflict with Windows.h */

#include <cstdint>
#include <iostream>
#include <string_view>

enum class MessageType
{
	HANDSHAKE	= 0,
	LOGIN		= 1,
	ERROR		= 2
};

enum class ErrorType : uint32_t
{
	LOGIN_FAILED = 0
};

/*!
 * \brief Header of all network messages
 */
#pragma pack(push, 1)
class Message
{
	public:
		Message() noexcept = delete;

		uint32_t GetSequenceID() const noexcept
		{
			return ntohl(m_sequence_id);
		}

		uint32_t GetSize() const noexcept
		{
			return ntohl(m_size);
		}

		MessageType GetType() const noexcept
		{
			return static_cast<MessageType>(ntohl(m_type));
		}

		void SetSequenceID(uint32_t sequence_id) noexcept
		{
			m_sequence_id = htonl(sequence_id);
		}

	protected:
		Message(uint32_t size, MessageType type) noexcept :
		m_sequence_id(0),
		m_size(htonl(size)),
		m_type(htonl(static_cast<uint32_t>(type)))
		{}

	private:
		uint32_t m_sequence_id;	/* Used for UDP message */
		uint32_t m_size;		/* Total size of the message */
		uint32_t m_type;		/* Type of the message */
};
#pragma pack(pop)

static constexpr size_t ERROR_FIELD_SIZE = 256;	/* Login field size */
/*!
 * \brief Contains an error code and a message if needed
 */
#pragma pack(push, 1)
class ErrorMessage : public Message
{
public:
	ErrorMessage(ErrorType error, std::string_view message) :
		Message(sizeof(ErrorMessage), MessageType::ERROR),
		m_error(htonl(static_cast<uint32_t>(error))),
		m_message()
	{
		std::copy(message.begin(), message.end(), m_message.begin());
	}

	std::string_view GetMessage() const noexcept
	{
		return std::string_view(reinterpret_cast<const char*>(m_message.data()), strnlen_s(reinterpret_cast<const char*>(m_message.data()), ERROR_FIELD_SIZE));
	}

	ErrorType GetError() const noexcept
	{
		return static_cast<ErrorType>(ntohl(m_error));
	}

private:
	uint32_t m_error;									/* Error code */
	std::array<uint8_t, ERROR_FIELD_SIZE> m_message;	/* Error message */
};
#pragma pack(pop)

/*!
 * \brief Message that is needed to link UDP and TCP connexion in a same client
 */
#pragma pack(push, 1)
class HandshakeMessage : public Message
{
public:
	HandshakeMessage(uint32_t unique_id) :
		Message(sizeof(HandshakeMessage), MessageType::HANDSHAKE),
		m_unique_id(htonl(unique_id))
	{}

	uint32_t GetUniqueID() const noexcept
	{
		return ntohl(m_unique_id);
	}

private:
	uint32_t m_unique_id;	/* Unique ID in order to link UDP and TCP connections */
};
#pragma pack(pop)

static constexpr size_t LOGIN_USERNAME_FIELD_SIZE = 32;	/* Login field size */
static constexpr size_t LOGIN_PASSWORD_FIELD_SIZE = 64;	/* Login field size */
/*!
 * \brief Contains the username and password of the user
 */
#pragma pack(push, 1)
class LoginMessage : public Message
{
public:
	LoginMessage(std::string_view username, std::string_view password) :
		Message(sizeof(LoginMessage), MessageType::LOGIN),
		m_username(),
		m_password()
	{
		std::copy(username.begin(), username.end(), m_username.begin());
		std::copy(password.begin(), password.end(), m_password.begin());
	}

	std::string_view GetUsername() const noexcept
	{
		return std::string_view(reinterpret_cast<const char*>(m_username.data()), strnlen_s(reinterpret_cast<const char*>(m_username.data()), LOGIN_USERNAME_FIELD_SIZE));
	}

	std::string_view GetPassword() const noexcept
	{
		return std::string_view(reinterpret_cast<const char*>(m_password.data()), strnlen_s(reinterpret_cast<const char*>(m_password.data()), LOGIN_PASSWORD_FIELD_SIZE));
	}

private:
	std::array<uint8_t, LOGIN_USERNAME_FIELD_SIZE> m_username;	/* User username */
	std::array<uint8_t, LOGIN_PASSWORD_FIELD_SIZE> m_password;	/* User password */
};
#pragma pack(pop)

#endif