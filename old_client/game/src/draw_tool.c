#include "raylib.h"
#include "game.h"
#include "draw_tool.h"

extern t_game_infos game;

/*
 * \brief draw a text with the default font at the wanted position with the wanted anchor
 *
 * \param[in] text is the text to write
 * \param[in] position is the position where the text has to be draw
 * \param[in] text_size is the size we want for the text
 * \param[in] color is the color of the text
 * \param[in] anchor_mode is the mode of the anchor
 */
static void draw_text(const char* text, Vector2 position, float text_size, Color color, t_anchor_mode mode)
{
	Font* font = get_map_element(&game.assets, INDEX_UI_FONT);
	Vector2 measure_to_substract = MeasureTextEx(*font, text, text_size * game.settings.scale, 2.0f * game.settings.scale);

	if (mode & PRIVATE_ANCHOR_HORIZONTAL_LEFT)
		measure_to_substract.x = 0;
	else if (mode & PRIVATE_ANCHOR_HORIZONTAL_CENTER)
		measure_to_substract.x *= 0.5f;

	if (mode & PRIVATE_ANCHOR_VERTICAL_TOP)
		measure_to_substract.y = 0;
	else if (mode & PRIVATE_ANCHOR_VERTICAL_CENTER)
		measure_to_substract.y *= 0.5f;

	DrawTextEx(*font, text, (Vector2) { position.x* game.settings.scale - measure_to_substract.x, position.y* game.settings.scale - measure_to_substract.y }, text_size * game.settings.scale, 2.0f * game.settings.scale, color);
}

void draw(t_draw_info *info)
{
	if (info->type == DRAW_TEXTURE)
	{
		t_draw_texture *texture_to_draw = info->data;
		Rectangle resized_destination;
		resized_destination.x = texture_to_draw->position.x * game.settings.scale;
		resized_destination.y = texture_to_draw->position.y * game.settings.scale;
		resized_destination.width = texture_to_draw->texture->width * texture_to_draw->scale * game.settings.scale;
		resized_destination.height = texture_to_draw->texture->height * texture_to_draw->scale * game.settings.scale;
		DrawTexturePro(*texture_to_draw->texture, texture_to_draw->src_in_texture, resized_destination, (Vector2) { 0, 0 }, texture_to_draw->rotation, texture_to_draw->color);
	}
	else if (info->type == DRAW_TEXT)
	{
		t_draw_text *text_to_draw = info->data;
		draw_text(text_to_draw->text, text_to_draw->position, text_to_draw->text_size, text_to_draw->color, text_to_draw->mode);
	}
}