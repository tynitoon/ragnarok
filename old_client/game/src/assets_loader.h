#ifndef ASSETS_LOADER_H
#define ASSETS_LOADER_H

#include "map.h"

typedef enum    e_asset_group
{
	GROUP_UI		= 0,
	GROUP_MAIN_MENU,
	GROUP_MAP_01
}				t_asset_group;

typedef enum    e_asset_index
{
	INDEX_UI_FONT = 0,
	INDEX_UI_BUTTON_BIG,
	INDEX_MAIN_MENU_MUSIC,
	INDEX_MAIN_MENU_BACKGROUND,
}				t_asset_index;

void load_group_assets(t_map* assets, t_asset_group group);
void unload_group_assets(t_map* assets, t_asset_group group);

#endif
