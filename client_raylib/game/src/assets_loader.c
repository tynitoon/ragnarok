#include <string.h>

#include "raylib.h"
#include "assets_loader.h"
#include "map.h"

/* Loaders */
static Font* load_font(char* path)
{
	Font*	pointer = get_memory(sizeof(Font));
	Font	tmp		= LoadFont(path);

	memcpy(pointer, &tmp, sizeof(Font));

	return pointer;
}

static Music* load_music(char* path)
{
	Music*	pointer = get_memory(sizeof(Music));
	Music	tmp		= LoadMusicStream(path);

	memcpy(pointer, &tmp, sizeof(Music));

	return pointer;
}

static Sound* load_sound(char* path)
{
	Sound*	pointer = get_memory(sizeof(Sound));
	Sound	tmp		= LoadSound(path);

	memcpy(pointer, &tmp, sizeof(Sound));

	return pointer;
}

static Image* load_image(char* path)
{
	Image*	pointer	= get_memory(sizeof(Image));
	Image	tmp		= LoadImage(path);

	memcpy(pointer, &tmp, sizeof(Image));

	return pointer;
}

/* Unloaders */
static void unload_font(void* data)
{
	UnloadFont(*(Font*)data);
	free_memory(data);
}

static void unload_music(void* data)
{
	UnloadMusicStream(*(Music*)data);
	free_memory(data);
}

static void unload_sound(void* data)
{
	UnloadSound(*(Sound*)data);
	free_memory(data);
}

static void unload_image(void* data)
{
	UnloadImage(*(Image*)data);
	free_memory(data);
}

/* Group Loaders */
static void load_UI(t_map* assets)
{
	add_map_element(assets, INDEX_UI_FONT,				load_font("resources/UI/font.png"));
	add_map_element(assets, INDEX_UI_BUTTON_BIG,		load_image("resources/UI/button_big.png"));
}

static void load_main_menu(t_map* assets)
{
	add_map_element(assets, INDEX_MAIN_MENU_BACKGROUND, load_image("resources/main_menu/background.png"));
	add_map_element(assets, INDEX_MAIN_MENU_MUSIC,		load_music("resources/main_menu/music.png"));
}

/* Group Unloaders */
static void unload_UI(t_map* assets)
{
	unload_font(remove_map_element(assets, INDEX_UI_FONT));
	unload_image(remove_map_element(assets, INDEX_UI_BUTTON_BIG));
}

static void unload_main_menu(t_map* assets)
{
	unload_image(remove_map_element(assets, INDEX_MAIN_MENU_BACKGROUND));
	unload_music(remove_map_element(assets, INDEX_MAIN_MENU_MUSIC));
}

/* Assets Loader Functions */
void load_group_assets(t_map* assets, t_asset_group group)
{
	switch (group)
	{
		case GROUP_UI:
			load_UI(assets);
			break;
		case GROUP_MAIN_MENU:
			load_main_menu(assets);
			break;
		default:
			break;
	}
}

void unload_group_assets(t_map* assets, t_asset_group group)
{
	switch (group)
	{
		case GROUP_UI:
			unload_UI(assets);
			break;
		case GROUP_MAIN_MENU:
			unload_main_menu(assets);
			break;
		default:
			break;
	}
}