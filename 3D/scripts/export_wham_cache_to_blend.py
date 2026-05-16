"""Export a prepared WHAM mesh cache to a Blender .blend file.

Run with Blender after prepare_wham_blend_cache.py has produced the cache:

    blender --background --python scripts/export_wham_cache_to_blend.py -- \
      --cache results/wham_visualizations/video_2/wham_blend_cache.npz \
      --output results/wham_visualizations/video_2/wham_output_mesh.blend

WHAM/SMPL vertices are meters, Y-up. Blender is Z-up, so coordinates are mapped
as (x, y, z) -> (x, -z, y).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


PALETTE = np.asarray(
    [
        [0.18, 0.78, 0.28, 1.0],
        [0.96, 0.45, 0.18, 1.0],
        [0.22, 0.47, 0.95, 1.0],
        [0.88, 0.22, 0.72, 1.0],
        [0.95, 0.82, 0.24, 1.0],
        [0.18, 0.78, 0.82, 1.0],
        [0.85, 0.85, 0.85, 1.0],
    ],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Export WHAM SMPL mesh animation to .blend.")
    parser.add_argument("--cache", required=True, help="Prepared .npz cache from prepare_wham_blend_cache.py.")
    parser.add_argument("--output", required=True, help="Output .blend path.")
    parser.add_argument("--decimate-viewer", action="store_true", help="Add viewport-only decimate modifiers.")
    return parser.parse_args(argv)


def source_to_blender_array(vertices: np.ndarray) -> np.ndarray:
    converted = np.empty_like(vertices, dtype=np.float32)
    converted[..., 0] = vertices[..., 0]
    converted[..., 1] = -vertices[..., 2]
    converted[..., 2] = vertices[..., 1]
    return converted


def source_to_blender_vec(point: np.ndarray) -> Vector:
    return Vector((float(point[0]), float(-point[2]), float(point[1])))


def clean_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def make_material(name: str, color: np.ndarray) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.diffuse_color = tuple(float(v) for v in color)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = material.diffuse_color
        bsdf.inputs["Roughness"].default_value = 0.58
    return material


def make_collection(name: str) -> bpy.types.Collection:
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    return collection


def link_to_collection(obj: bpy.types.Object, collection: bpy.types.Collection) -> None:
    collection.objects.link(obj)
    for existing in list(obj.users_collection):
        if existing != collection:
            existing.objects.unlink(obj)


def subject_sort_key(subject_id: str) -> tuple[int, str]:
    if subject_id == "character_A":
        return (0, subject_id)
    if subject_id == "character_B":
        return (1, subject_id)
    return (int(subject_id), subject_id) if subject_id.isdigit() else (10_000, subject_id)


def subject_color(subject_id: str) -> np.ndarray:
    if subject_id == "character_A":
        return np.asarray([0.18, 0.78, 0.28, 1.0], dtype=np.float32)
    if subject_id == "character_B":
        return np.asarray([0.96, 0.45, 0.18, 1.0], dtype=np.float32)
    try:
        index = int(subject_id)
    except ValueError:
        index = abs(hash(subject_id))
    return PALETTE[index % len(PALETTE)]


def iter_action_fcurves(action):
    if action is None:
        return
    if hasattr(action, "fcurves"):
        for fcurve in action.fcurves:
            yield fcurve
        return
    for layer in getattr(action, "layers", []):
        for strip in getattr(layer, "strips", []):
            for channelbag in getattr(strip, "channelbags", []):
                for fcurve in getattr(channelbag, "fcurves", []):
                    yield fcurve


def set_linear_interpolation(id_data) -> None:
    if not id_data.animation_data or not id_data.animation_data.action:
        return
    for fcurve in iter_action_fcurves(id_data.animation_data.action):
        if fcurve.data_path != "eval_time":
            continue
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = "LINEAR"


def set_constant_interpolation(obj: bpy.types.Object) -> None:
    if not obj.animation_data or not obj.animation_data.action:
        return
    for fcurve in iter_action_fcurves(obj.animation_data.action):
        if fcurve.data_path not in {"hide_viewport", "hide_render"}:
            continue
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = "CONSTANT"


def animate_visibility(obj: bpy.types.Object, first_scene_frame: int, last_scene_frame: int, scene_start: int, scene_end: int) -> None:
    if scene_start < first_scene_frame:
        obj.hide_viewport = True
        obj.hide_render = True
        obj.keyframe_insert(data_path="hide_viewport", frame=scene_start)
        obj.keyframe_insert(data_path="hide_render", frame=scene_start)
        obj.keyframe_insert(data_path="hide_viewport", frame=max(scene_start, first_scene_frame - 1))
        obj.keyframe_insert(data_path="hide_render", frame=max(scene_start, first_scene_frame - 1))

    obj.hide_viewport = False
    obj.hide_render = False
    obj.keyframe_insert(data_path="hide_viewport", frame=first_scene_frame)
    obj.keyframe_insert(data_path="hide_render", frame=first_scene_frame)
    obj.keyframe_insert(data_path="hide_viewport", frame=last_scene_frame)
    obj.keyframe_insert(data_path="hide_render", frame=last_scene_frame)

    if last_scene_frame < scene_end:
        obj.hide_viewport = True
        obj.hide_render = True
        obj.keyframe_insert(data_path="hide_viewport", frame=last_scene_frame + 1)
        obj.keyframe_insert(data_path="hide_render", frame=last_scene_frame + 1)

    set_constant_interpolation(obj)


def set_location_interpolation(obj: bpy.types.Object) -> None:
    if not obj.animation_data or not obj.animation_data.action:
        return
    for fcurve in iter_action_fcurves(obj.animation_data.action):
        if fcurve.data_path != "location":
            continue
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = "LINEAR"


def animate_object_location(obj: bpy.types.Object, frame_ids: np.ndarray, locations: np.ndarray) -> None:
    for frame_id, location in zip(frame_ids, locations):
        obj.location = tuple(float(value) for value in location)
        obj.keyframe_insert(data_path="location", frame=int(frame_id))
    set_location_interpolation(obj)


def add_absolute_shape_keys(
    obj: bpy.types.Object,
    frame_ids: np.ndarray,
    vertices_blender: np.ndarray,
) -> None:
    basis = obj.shape_key_add(name=f"frame_{int(frame_ids[0]):04d}", from_mix=False)
    obj.data.shape_keys.use_relative = False
    basis.data.foreach_set("co", vertices_blender[0].reshape(-1))

    for local_idx in range(1, len(frame_ids)):
        key = obj.shape_key_add(name=f"frame_{int(frame_ids[local_idx]):04d}", from_mix=False)
        key.data.foreach_set("co", vertices_blender[local_idx].reshape(-1))
        if local_idx % 250 == 0:
            print(f"  shape keys: {local_idx}/{len(frame_ids)}")

    shape_keys = obj.data.shape_keys
    # Blender spaces absolute shape keys at eval_time 0, 10, 20... and makes
    # ShapeKey.frame read-only in recent versions. Animate eval_time in that
    # native spacing; WHAM frame_ids are contiguous within each subject segment.
    last_eval_time = float((len(frame_ids) - 1) * 10.0)
    shape_keys.eval_time = 0.0
    shape_keys.keyframe_insert(data_path="eval_time", frame=int(frame_ids[0]))
    shape_keys.eval_time = last_eval_time
    shape_keys.keyframe_insert(data_path="eval_time", frame=int(frame_ids[-1]))
    set_linear_interpolation(shape_keys)


def make_subject_mesh(
    subject_id: str,
    frame_ids: np.ndarray,
    vertices: np.ndarray,
    roots: np.ndarray,
    faces: np.ndarray,
    material: bpy.types.Material,
    collection: bpy.types.Collection,
    scene_start: int,
    scene_end: int,
    decimate_viewer: bool,
) -> bpy.types.Object:
    vertices_blender = source_to_blender_array(vertices)
    locations = source_to_blender_array(roots)
    local_vertices = vertices_blender - locations[:, None, :]
    first_vertices = local_vertices[0]
    mesh = bpy.data.meshes.new(f"wham_subject_{subject_id}_mesh")
    mesh.from_pydata(first_vertices.tolist(), [], faces.tolist())
    mesh.update()
    obj = bpy.data.objects.new(f"WHAM_subject_{subject_id}", mesh)
    obj.data.materials.append(material)
    link_to_collection(obj, collection)

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.shade_smooth()
    obj.select_set(False)

    if decimate_viewer:
        decimate = obj.modifiers.new("viewport_decimate", "DECIMATE")
        decimate.ratio = 0.5
        decimate.show_render = False

    add_absolute_shape_keys(obj, frame_ids, local_vertices)
    animate_object_location(obj, frame_ids, locations)
    animate_visibility(
        obj,
        first_scene_frame=int(frame_ids[0]),
        last_scene_frame=int(frame_ids[-1]),
        scene_start=scene_start,
        scene_end=scene_end,
    )
    return obj


def converted_bounds(bounds_min: np.ndarray, bounds_max: np.ndarray) -> tuple[Vector, Vector]:
    corners = []
    for x in (bounds_min[0], bounds_max[0]):
        for y in (bounds_min[1], bounds_max[1]):
            for z in (bounds_min[2], bounds_max[2]):
                corners.append(source_to_blender_vec(np.asarray([x, y, z], dtype=np.float32)))
    out_min = Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners)))
    out_max = Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners)))
    return out_min, out_max


def make_floor(bounds_min: Vector, bounds_max: Vector) -> None:
    material = make_material("matte_ground_material", np.asarray([0.42, 0.44, 0.41, 1.0], dtype=np.float32))
    center = (bounds_min + bounds_max) * 0.5
    span_x = max(2.0, bounds_max.x - bounds_min.x)
    span_y = max(2.0, bounds_max.y - bounds_min.y)
    size = max(span_x, span_y) + 1.5
    bpy.ops.mesh.primitive_plane_add(size=size, location=(center.x, center.y, 0.0))
    floor = bpy.context.object
    floor.name = "WHAM_floor_z0"
    floor.data.materials.append(material)
    wire = floor.modifiers.new("floor_grid", "WIREFRAME")
    wire.thickness = 0.004
    wire.use_even_offset = True


def make_lights(bounds_min: Vector, bounds_max: Vector) -> None:
    center = (bounds_min + bounds_max) * 0.5

    bpy.ops.object.light_add(type="AREA", location=(center.x - 2.0, center.y - 3.0, center.z + 5.0))
    key = bpy.context.object
    key.name = "WHAM_soft_key_light"
    key.data.energy = 650.0
    key.data.size = 5.0

    bpy.ops.object.light_add(type="POINT", location=(center.x + 2.5, center.y + 2.5, center.z + 2.5))
    fill = bpy.context.object
    fill.name = "WHAM_fill_light"
    fill.data.energy = 75.0


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def make_camera_and_view_target(bounds_min: Vector, bounds_max: Vector) -> None:
    center = (bounds_min + bounds_max) * 0.5
    span = bounds_max - bounds_min
    radius = max(float(span.x), float(span.y), float(span.z), 2.0)
    target = Vector((center.x, center.y, max(center.z, 0.9)))

    camera_location = Vector((center.x, center.y + radius * 2.1, center.z + radius * 0.75 + 0.6))
    bpy.ops.object.camera_add(location=camera_location)
    camera = bpy.context.object
    camera.name = "WHAM_overview_camera"
    camera.data.lens = 35.0
    camera.data.clip_start = 0.001
    camera.data.clip_end = 10000.0
    camera.data.dof.use_dof = False
    look_at(camera, target)
    bpy.context.scene.camera = camera

    for area in getattr(bpy.context.screen, "areas", []):
        if area.type != "VIEW_3D":
            continue
        for space in area.spaces:
            if space.type == "VIEW_3D":
                space.clip_start = 0.001
                space.clip_end = 10000.0


def set_scene_fps(scene: bpy.types.Scene, fps_value: float) -> None:
    fps_value = float(fps_value) if np.isfinite(fps_value) and fps_value > 0.0 else 30.0
    fps_int = max(1, int(round(fps_value)))
    scene.render.fps = fps_int
    scene.render.fps_base = float(fps_int) / fps_value


def make_notes(metadata: dict, output_path: Path) -> None:
    text = bpy.data.texts.new("wham_blend_export_notes")
    body = "\n".join(
        [
            "WHAM visualization mesh export",
            f"Output: {output_path.name}",
            f"WHAM source: {metadata.get('wham_output', '')}",
            "Coordinates: WHAM/SMPL meters Y-up -> Blender meters Z-up, (x, y, z) -> (x, -z, y)",
            f"Subjects: {', '.join(metadata.get('subjects', []))}",
        ]
    )
    text.write(body)


def export_blend(cache_path: Path, output_path: Path, decimate_viewer: bool) -> None:
    clean_scene()
    cache = np.load(cache_path, allow_pickle=False)
    subject_ids = [str(item) for item in cache["subject_ids"].tolist()]
    subject_ids = sorted(subject_ids, key=subject_sort_key)
    faces = np.asarray(cache["faces"], dtype=np.int32)
    metadata = json.loads(str(cache["metadata_json"].item()))

    all_frame_ids = []
    for subject_id in subject_ids:
        all_frame_ids.extend(np.asarray(cache[f"frame_ids__{subject_id}"], dtype=np.int64).tolist())
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    set_scene_fps(scene, float(metadata.get("fps", 30.0)))
    scene.frame_start = int(min(all_frame_ids))
    scene.frame_end = int(max(all_frame_ids))

    for subject_id in subject_ids:
        print(f"Exporting WHAM subject {subject_id}")
        collection = make_collection(f"WHAM_subject_{subject_id}")
        material = make_material(f"WHAM_subject_{subject_id}_material", subject_color(subject_id))
        frame_ids = np.asarray(cache[f"frame_ids__{subject_id}"], dtype=np.int64)
        vertices = np.asarray(cache[f"verts__{subject_id}"], dtype=np.float32)
        roots = np.asarray(cache[f"roots__{subject_id}"], dtype=np.float32)
        make_subject_mesh(
            subject_id=subject_id,
            frame_ids=frame_ids,
            vertices=vertices,
            roots=roots,
            faces=faces,
            material=material,
            collection=collection,
            scene_start=scene.frame_start,
            scene_end=scene.frame_end,
            decimate_viewer=decimate_viewer,
        )

    bounds_min, bounds_max = converted_bounds(cache["bounds_min"], cache["bounds_max"])
    make_camera_and_view_target(bounds_min, bounds_max)
    make_notes(metadata, output_path)

    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.frame_set(scene.frame_start)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path))


def main() -> None:
    args = parse_args()
    export_blend(Path(args.cache).resolve(), Path(args.output).resolve(), args.decimate_viewer)


if __name__ == "__main__":
    main()
