import os
import sys
import subprocess
import pathlib

def compile(shader_files, *, force_recompile=False, is_uv_buffer=False):
    glslang_cmd = "glslangValidator"

    up_to_date = []

    for shader in shader_files:
        output = f"{shader}{'.uvbuf' if is_uv_buffer else ''}.spv"
        cmd = [glslang_cmd, "-V", "-g"] + (["-DUV_BUFFER"] if is_uv_buffer else []) + [shader, "-o", output]

        if force_recompile or not os.path.exists(output) or os.path.getmtime(shader) > os.path.getmtime(output):
            subprocess.run(cmd)
        else:
            up_to_date.append(shader)

    if up_to_date:
        print(f"Up to date: {', '.join(up_to_date)}")


if __name__ == '__main__':
    force_recompile = "-f" in sys.argv

    shader_list = [
        "gbuffer.vert", "gbuffer.frag",
        "resolve.vert", "resolve.frag",
        "depth_only.vert", "genbrdflut.vert",
        "genbrdflut.frag", "filtercube.vert",
        "irradiancecube.frag", "prefilterenvmap.frag",
        "postfx.frag"
    ]

    compile(shader_list, force_recompile=force_recompile)

    shader_list = [
        "gbuffer.frag", "resolve.frag"
    ]

    compile(shader_list, force_recompile=force_recompile, is_uv_buffer=True)
