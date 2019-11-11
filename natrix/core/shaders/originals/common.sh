
uvec4 GetNeighbours(ivec2 pos, ivec2 size)
{
    uvec4 result;
    int maxX = size.x - 1;
    int maxY = size.y - 1;
    result.x = uint(pos.y) * _Size.x + uint(clamp(pos.x - 1, 0, maxX));
    result.y = uint(pos.y) * _Size.x + uint(clamp(pos.x + 1, 0, maxX));
    result.z = uint(clamp(pos.y - 1, 0, maxY) * size.x + pos.x);
    result.w = uint(clamp(pos.y + 1, 0, maxY) * size.x + pos.x);
    return result;
}
