#version 450
#extension GL_EXT_debug_printf : enable

layout(push_constant) uniform PushConsts {
	layout (offset = 0) mat4 mvp;
} pushConsts;

layout (location = 0) out vec3 outUVW;

out gl_PerVertex {
	vec4 gl_Position;
};

vec3 getVertex(int i) {
	return vec3(
		(i & 4) != 0 ? 1 : -1,
		(i & 2) != 0 ? 1 : -1,
		(i & 1) != 0 ? 1 : -1);
}

int chooseThree(int i, int a, int b, int c) {
	if (i == 0)
		return a;
	if (i == 1)
		return b;
	if (i == 2)
		return c;
	return a;
}

// Broken
vec3 getVertexCube() {
	const int inst = gl_VertexIndex / 3;
	const int inst_vertex = gl_VertexIndex % 3;

	int v1 = 1 << (inst / 3);
	int v2 = v1 == 4 ? 1 : v1 << 1;

	switch (inst % 4) {
		case 0: return getVertex(chooseThree(inst_vertex, 0, v1, v2));
		case 1: return getVertex(chooseThree(inst_vertex, v1 + v2, v2, v1));
		case 2: return getVertex(chooseThree(inst_vertex, 7, v2, 7 - v1));
		case 3: return getVertex(chooseThree(inst_vertex, 7 - (v1 + v2), 7 - v1, 7 - v2));
	}

	return vec3(0);
}

vec3 getVertexSimplex(int i) {
	switch (i) {
		case 0: return vec3(0, 3, 0);
		case 1: return vec3(3, -1, 0);
		case 2: return vec3(-1, -1, 3);
		case 3: return vec3(-1, -1, -3);
	}
	return vec3(0);
}

vec3 getVertexSimplex() {
	const int inst = gl_VertexIndex / 3;
	const int inst_vertex = gl_VertexIndex % 3;

	switch (inst) {
		case 0: return getVertexSimplex(chooseThree(inst_vertex, 0, 1, 2));
		case 1: return getVertexSimplex(chooseThree(inst_vertex, 0, 2, 3));
		case 2: return getVertexSimplex(chooseThree(inst_vertex, 0, 3, 1));
		case 3: return getVertexSimplex(chooseThree(inst_vertex, 3, 2, 1));
	}

	return vec3(0);
}

void main() 
{
	outUVW = getVertexSimplex();
		// debugPrintfEXT("point %d: %v3f\n", gl_VertexIndex, outUVW.xyz);
	gl_Position = pushConsts.mvp * vec4(getVertexSimplex(), 1.0);
}
