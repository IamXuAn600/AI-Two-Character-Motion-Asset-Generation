"""Add small red head markers for in-between occlusion spans.

The marker is a mesh torus animated above the occluded character's head during
the missing-frame interval. It does not freeze or hide any character.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser(description="Add red head markers for occlusion in-between events.")
    parser.add_argument("--input", required=True, help="Input .blend path.")
    parser.add_argument("--cache", required=True, help="Final cache used to export the blend.")
    parser.add_argument("--events-json", required=True, help="occlusion_inbetween_events.json.")
    parser.add_argument("--output", required=True, help="Output .blend path.")
    parser.add_argument("--radius", type=float, default=0.10)
    parser.add_argument("--tube-radius", type=float, default=0.012)
    parser.add_argument("--height-offset", type=float, default=0.40)
    return parser.parse_args(argv)


def source_to_blender_array(vertices: np.ndarray) -> np.ndarray:
    converted = np.empty_like(vertices, dtype=np.float32)
    converted[..., 0] = vertices[..., 0]
    converted[..., 1] = -vertices[..., 2]
    converted[..., 2] = vertices[..., 1]
    return converted


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


def set_interpolation(obj: bpy.types.Object, data_path: str, interpolation: str) -> None:
    if not obj.animation_data or not obj.animation_data.action:
        return
    for fcurve in iter_action_fcurves(obj.animation_data.action):
        if fcurve.data_path != data_path:
            continue
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = interpolation


def make_marker_material() -> bpy.types.Material:
    material = bpy.data.materials.new("red_head_occlusion_marker_material")
    material.diffuse_color = (1.0, 0.02, 0.02, 1.0)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = material.diffuse_color
        bsdf.inputs["Emission Color"].default_value = (1.0, 0.0, 0.0, 1.0)
        bsdf.inputs["Emission Strength"].default_value = 5.0
        bsdf.inputs["Roughness"].default_value = 0.35
    return material


def make_collection(name: str) -> bpy.types.Collection:
    existing = bpy.data.collections.get(name)
    if existing is not None:
        for obj in list(existing.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        return existing
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    return collection


def link_to_collection(obj: bpy.types.Object, collection: bpy.types.Collection) -> None:
    collection.objects.link(obj)
    for existing in list(obj.users_collection):
        if existing != collection:
            existing.objects.unlink(obj)


def head_marker_position(vertices_source: np.ndarray, height_offset: float) -> np.ndarray:
    vertices = source_to_blender_array(vertices_source)
    z_values = vertices[:, 2]
    finite = np.isfinite(z_values)
    if not np.any(finite):
        return np.zeros(3, dtype=np.float32)
    top_z = float(np.max(z_values[finite]))
    threshold = float(np.percentile(z_values[finite], 98.5))
    top_vertices = vertices[np.isfinite(z_values) & (z_values >= threshold)]
    if len(top_vertices) == 0:
        top_vertices = vertices[np.asarray(z_values) == top_z]
    xy = np.mean(top_vertices[:, :2], axis=0)
    return np.asarray([float(xy[0]), float(xy[1]), top_z + height_offset], dtype=np.float32)


def animate_visibility(obj: bpy.types.Object, start_frame: int, end_frame: int, scene_start: int, scene_end: int) -> None:
    obj.hide_viewport = True
    obj.hide_render = True
    obj.keyframe_insert(data_path="hide_viewport", frame=scene_start)
    obj.keyframe_insert(data_path="hide_render", frame=scene_start)
    obj.keyframe_insert(data_path="hide_viewport", frame=max(scene_start, start_frame - 1))
    obj.keyframe_insert(data_path="hide_render", frame=max(scene_start, start_frame - 1))

    obj.hide_viewport = False
    obj.hide_render = False
    obj.keyframe_insert(data_path="hide_viewport", frame=start_frame)
    obj.keyframe_insert(data_path="hide_render", frame=start_frame)
    obj.keyframe_insert(data_path="hide_viewport", frame=end_frame)
    obj.keyframe_insert(data_path="hide_render", frame=end_frame)

    if end_frame < scene_end:
        obj.hide_viewport = True
        obj.hide_render = True
        obj.keyframe_insert(data_path="hide_viewport", frame=end_frame + 1)
        obj.keyframe_insert(data_path="hide_render", frame=end_frame + 1)

    set_interpolation(obj, "hide_viewport", "CONSTANT")
    set_interpolation(obj, "hide_render", "CONSTANT")


def create_marker(
    event: dict,
    cache: np.lib.npyio.NpzFile,
    material: bpy.types.Material,
    collection: bpy.types.Collection,
    scene_start: int,
    scene_end: int,
    radius: float,
    tube_radius: float,
    height_offset: float,
) -> bool:
    role = str(event["role"])
    verts_key = f"verts__{role}"
    if verts_key not in cache.files:
        print(f"Skipping event {event.get('event_id')}: missing {verts_key}")
        return False
    vertices = np.asarray(cache[verts_key], dtype=np.float32)
    start = int(event.get("prediction_trigger_frame", event["occlusion_start_frame"]))
    end = int(event["occlusion_end_frame"])
    if end <= start:
        return False
    last_missing = min(end - 1, len(vertices) - 1)
    start = max(0, min(start, last_missing))

    first_location = head_marker_position(vertices[start], height_offset)
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=32,
        ring_count=16,
        radius=float(radius),
        location=tuple(float(v) for v in first_location),
    )
    marker = bpy.context.object
    marker.name = f"HeadOcclusionDot_{int(event['event_id']):03d}_{role}"
    marker.data.name = f"head_occlusion_dot_{int(event['event_id']):03d}_{role}_mesh"
    marker.data.materials.append(material)
    marker.show_in_front = True
    try:
        bpy.ops.object.shade_smooth()
    except Exception:
        pass
    link_to_collection(marker, collection)

    for frame_idx in range(start, last_missing + 1):
        location = head_marker_position(vertices[frame_idx], height_offset)
        marker.location = tuple(float(v) for v in location)
        marker.keyframe_insert(data_path="location", frame=frame_idx)
    set_interpolation(marker, "location", "LINEAR")

    animate_visibility(marker, start, last_missing, scene_start, scene_end)
    return True


def main() -> None:
    args = parse_args()
    bpy.ops.wm.open_mainfile(filepath=str(Path(args.input).resolve()))
    cache = np.load(Path(args.cache).resolve(), allow_pickle=True)
    payload = json.loads(Path(args.events_json).read_text(encoding="utf-8"))
    events = payload.get("events", payload if isinstance(payload, list) else [])

    scene = bpy.context.scene
    material = make_marker_material()
    collection = make_collection("Head_occlusion_markers")
    created = 0
    for event in events:
        if create_marker(
            event,
            cache,
            material,
            collection,
            scene.frame_start,
            scene.frame_end,
            args.radius,
            args.tube_radius,
            args.height_offset,
        ):
            created += 1

    visible_starts = [
        int(event.get("prediction_trigger_frame", event["occlusion_start_frame"]))
        for event in events
        if event.get("role")
    ]
    if visible_starts:
        scene.frame_set(max(scene.frame_start, min(visible_starts)))

    note = bpy.data.texts.new("head_occlusion_marker_events")
    note.write(json.dumps({"marker_count": created, "events_json": str(Path(args.events_json).resolve())}, indent=2))
    bpy.ops.wm.save_as_mainfile(filepath=str(Path(args.output).resolve()))
    print(f"Added {created} red head occlusion markers")


if __name__ == "__main__":
    main()
