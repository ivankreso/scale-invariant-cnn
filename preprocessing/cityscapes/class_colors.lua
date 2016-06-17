
local class_info = {}
class_info[0] = {'unlabeled',     {0,0,0},        0}
class_info[1] = {'road',          {128,64,128},   7}
class_info[2] = {'sidewalk',      {244,35,232},   8}
class_info[3] = {'building',      {70,70,70},     11}
class_info[4] = {'wall',          {102,102,156},  12}
class_info[5] = {'fence',         {190,153,153},  13}
class_info[6] = {'pole',          {153,153,153},  17}
class_info[7] = {'traffic light', {250,170,30},   19}
class_info[8] = {'traffic sign',  {220,220,0},    20}
class_info[9] = {'vegetation',    {107,142,35},   21}
class_info[10] = {'terrain',      {152,251,152},  22}
class_info[11] = {'sky',          {70,130,180},   23}
class_info[12] = {'person',       {220,20,60},    24}
class_info[13] = {'rider',        {255,0,0},      25}
class_info[14] = {'car',          {0,0,142},      26}
class_info[15] = {'truck',        {0,0,70},       27}
class_info[16] = {'bus',          {0,60,100},     28}
class_info[17] = {'train',        {0,80,100},     31}
class_info[18] = {'motorcycle',   {0,0,230},      32}
class_info[19] = {'bicycle',      {119,11,32},    33}

local class_color_map = {}

for i = 0, #class_info do
  local r = class_info[i][2][1]
  local g = class_info[i][2][2]
  local b = class_info[i][2][3]
  local key = string.format("%03d", r) .. string.format("%03d", g) .. string.format("%03d", b)
  class_color_map[key] = {i, class_info[i][1], class_info[i][3]}
end

return class_info, class_color_map, 19
