#ifndef MESSAGE_HPP
#define MESSAGE_HPP

#include <iostream>

#include <cstdint>

enum class MessageType
{
	HANDSHAKE = 0,
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

#endif