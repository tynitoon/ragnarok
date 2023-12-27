#include "raylib.h"
#include "draw_tool.h"

extern t_game_infos game;

void draw_text(const char* text, Vector2 position, float font_size, Color color, t_alignment_mode mode)
{
	Font* font = get_map_element(&game.assets, INDEX_UI_FONT);
	Vector2 measure_to_substract = MeasureTextEx(*font, text, font_size, 2.0f);

	if (mode & ALIGNEMENT_HORIZONTAL_LEFT)
		measure_to_substract.x = 0;
	else if (mode & ALIGNEMENT_HORIZONTAL_CENTER)
		measure_to_substract.x *= 0.5f;

	if (mode & ALIGNEMENT_VERTICAL_TOP)
		measure_to_substract.y = 0;
	else if (mode & ALIGNEMENT_HORIZONTAL_CENTER)
		measure_to_substract.y *= 0.5f;

	DrawTextEx(*font, text, (Vector2) { position.x* game.settings.scale - measure_to_substract.x, position.y* game.settings.scale - measure_to_substract.y }, font_size* game.settings.scale, 2.0f, color);
}