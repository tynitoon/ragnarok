#ifndef DRAW_TOOL_H
#define DRAW_TOOL_H

#include "assets_loader.h"

typedef enum							e_anchor_mode
{
	PRIVATE_ANCHOR_HORIZONTAL_LEFT		= (1 << 0),
	PRIVATE_ANCHOR_HORIZONTAL_CENTER	= (1 << 1),
	PRIVATE_ANCHOR_HORIZONTAL_RIGHT		= (1 << 2),
	PRIVATE_ANCHOR_VERTICAL_TOP			= (1 << 3),
	PRIVATE_ANCHOR_VERTICAL_CENTER		= (1 << 4),
	PRIVATE_ANCHOR_VERTICAL_BOTTOM		= (1 << 5),

	ANCHOR_LEFT_TOP					= PRIVATE_ANCHOR_HORIZONTAL_LEFT | PRIVATE_ANCHOR_VERTICAL_TOP,
	ANCHOR_CENTER_TOP				= PRIVATE_ANCHOR_HORIZONTAL_CENTER | PRIVATE_ANCHOR_VERTICAL_TOP,
	ANCHOR_RIGHT_TOP				= PRIVATE_ANCHOR_HORIZONTAL_RIGHT | PRIVATE_ANCHOR_VERTICAL_TOP,
	ANCHOR_LEFT_CENTER				= PRIVATE_ANCHOR_HORIZONTAL_LEFT | PRIVATE_ANCHOR_VERTICAL_CENTER,
	ANCHOR_CENTER					= PRIVATE_ANCHOR_HORIZONTAL_CENTER | PRIVATE_ANCHOR_VERTICAL_CENTER,
	ANCHOR_RIGHT_CENTER				= PRIVATE_ANCHOR_HORIZONTAL_RIGHT | PRIVATE_ANCHOR_VERTICAL_CENTER,
	ANCHOR_LEFT_BOTTOM				= PRIVATE_ANCHOR_HORIZONTAL_LEFT | PRIVATE_ANCHOR_VERTICAL_BOTTOM,
	ANCHOR_CENTER_BOTTOM			= PRIVATE_ANCHOR_HORIZONTAL_CENTER | PRIVATE_ANCHOR_VERTICAL_BOTTOM,
	ANCHOR_RIGHT_BOTTOM				= PRIVATE_ANCHOR_HORIZONTAL_RIGHT | PRIVATE_ANCHOR_VERTICAL_BOTTOM,
}										t_anchor_mode;

typedef enum		s_draw_type
{
	DRAW_TEXTURE = 0,
	DRAW_TEXT
}					t_draw_type;

typedef struct		s_draw_info
{
	t_draw_type		type;
	char			data[0];
}					t_draw_info;

typedef struct		s_draw_texture
{
	Texture2D*		texture;
	Rectangle		src_in_texture;
	Vector2			position;
	float			rotation;
	float			scale;
	Color			color;
}					t_draw_texture;

typedef struct		s_draw_text
{
	t_anchor_mode	mode;
	Vector2			position;
	float			text_size;
	Color			color;
	char			text[0];
}					t_draw_text;

/*
 * \brief draw a text of a texture depending of the t_draw_info
 *
 * \param[in] info is the container of all information needed to draw
 */
void draw(t_draw_info* info);

#endif /* DRAW_TOOL_H */
