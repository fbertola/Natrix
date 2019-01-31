#version 310 es

precision mediump float;

uniform sampler2D srcTex;
in vec2 texCoord;
out vec4 color;

void main() {
    float c = texture(srcTex, texCoord).x;
    color = vec4(c, c, c, 1.0);
}