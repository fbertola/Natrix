$input v_texcoord0

#include <bgfx_shader.sh>
#include "bgfx_compute.sh"

SAMPLER2D(s_texColor, 0);
BUFFER_RO(_VelocityIn, vec2, 1);

uniform float ArrowTileSize;
uniform vec2 VelocitySize;
uniform vec2 WindowSize;


vec2 _arrow_tile_center_coord(vec2 pos) {
	return (floor(pos / ArrowTileSize) + 0.5) * ArrowTileSize;
}

float _line(vec2 p, vec2 p1, vec2 p2) {
	vec2 center = (p1 + p2) * 0.5;
	float len = length(p2 - p1);
	vec2 dir = (p2 - p1) / len;
	vec2 rel_p = p - center;
	float dist1 = abs(dot(rel_p, vec2(dir.y, -dir.x)));
	float dist2 = abs(dot(rel_p, dir)) - 0.5*len;
	return max(dist1, dist2);
}

float _vector(vec2 p, vec2 v) {
    p -= _arrow_tile_center_coord(p);
	float mag_v = length(v), mag_p = length(p);

	if (mag_v > 0.0) {
		vec2 dir_v = v / mag_v;

		mag_v = clamp(mag_v, 0.0, ArrowTileSize * 0.5);

		v = dir_v * mag_v;

		float shaft = _line(p, v, -v);
		float head = min(_line(p, v, 0.4*v + 0.2*vec2(-v.y, v.x)),
		                 _line(p, v, 0.4*v + 0.2*vec2(v.y, -v.x)));

		return min(shaft, head);
	} else {
		return mag_p;
	}
}

vec2 _field(vec2 pos) {
	ivec2 zero = ivec2(0, 0);
	vec2 one = vec2(1.0, 1.0);
	vec2 fPos = (one - (pos / WindowSize)) * VelocitySize;
    ivec2 size_bounds = ivec2(VelocitySize.x - 1u, VelocitySize.y - 1u);
    ivec2 top_right = ivec2(clamp(ceil(fPos), vec2(zero), vec2(size_bounds)));
    ivec2 bottom_left = ivec2(clamp(floor(fPos), vec2(zero), vec2(size_bounds)));
    vec2 delta = fPos - vec2(bottom_left);
    vec2 lt = _VelocityIn[uint(top_right.y) * VelocitySize.x + uint(bottom_left.x)];
    vec2 rt = _VelocityIn[uint(top_right.y) * VelocitySize.x + uint(top_right.x)];
    vec2 lb = _VelocityIn[uint(bottom_left.y) * VelocitySize.x + uint(bottom_left.x)];
    vec2 rb = _VelocityIn[uint(bottom_left.y) * VelocitySize.x + uint(top_right.x)];
    vec2 h1 = mix(lt, rt, vec2(delta.x, 0));
    vec2 h2 = mix(lb, rb, vec2(delta.x, 0));
    return -1.0 * mix(h2, h1, vec2(delta.y, 0)) * (WindowSize / VelocitySize);
}

void main()
{
    float arrow_dist = _vector(gl_FragCoord.xy, _field(_arrow_tile_center_coord(gl_FragCoord.xy)) * ArrowTileSize * 0.4);
	gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0 - clamp(arrow_dist, 0.0, 1.0));
}