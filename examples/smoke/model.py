"""A supplied synthetic box: Blender owns geometry; mesh-jig owns export and rendering."""
import bpy


def build():
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0.5))
    obj = bpy.context.active_object
    obj.scale = (0.6, 0.4, 1.0)
    colors = obj.data.color_attributes.new(name="Color", type="FLOAT_COLOR", domain="CORNER")
    for face in obj.data.polygons:
        color = (0.7, 0.4, 0.05, 1.0) if face.normal.z > 0.5 else (0.05, 0.25, 0.6, 1.0)
        for loop in face.loop_indices:
            colors.data[loop].color = color
