#version 310 es

precision mediump float;

uniform sampler2D srcTex;
in vec2 texCoord;
out vec4 color;

void main() {
    float c = texture(srcTex, texCoord).x;
    vec4 a = vec4(c, 0.0, 0.0, 1.0);
    color = floor(a * vec4(1024.0))/vec4(1024.0);
}