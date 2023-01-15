vec2 encode_normal(vec3 n)
{
    float p = sqrt(n.z * 8 + 8);
    return vec2(n.xy / p + 0.5);
}

vec3 decode_normal(vec2 enc)
{
    vec2 fenc = enc * 4 - 2;
    float f = dot(fenc,fenc);
    float g = sqrt(1-f/4);
    vec3 n;
    n.xy = fenc*g;
    n.z = 1-f/2;
    return n;
}