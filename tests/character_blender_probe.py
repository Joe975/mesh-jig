"""Run inside Blender: independent importer/player witness for composed GLBs."""
import json
from pathlib import Path
import sys

import bpy
import numpy as np
from io_scene_gltf2.blender.imp.animation_utils import restore_animation_on_object

request=Path(sys.argv[sys.argv.index('--')+1])
data=json.loads(request.read_text(encoding='utf-8'))
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.context.scene.render.fps=24
bpy.ops.import_scene.gltf(filepath=data['glb'])
results=[]
for index,case in enumerate(data['cases']):
    for obj in bpy.data.objects:
        if obj.animation_data:
            obj.animation_data.action=None
            for track in obj.animation_data.nla_tracks:
                track.mute=True
        if obj.type=='ARMATURE':
            for bone in obj.pose.bones:
                bone.location=(0,0,0)
                bone.rotation_mode='QUATERNION'
                bone.rotation_quaternion=(1,0,0,0)
                bone.scale=(1,1,1)
    if case.get('clip'):
        for obj in bpy.data.objects:
            restore_animation_on_object(obj,case['clip'])
    else:
        found=set()
        for obj in bpy.data.objects:
            if not obj.animation_data: continue
            for track in obj.animation_data.nla_tracks:
                if track.name in case.get('layers',[]):
                    found.add(track.name);track.mute=False
                    for strip in track.strips:
                        strip.blend_type='REPLACE';strip.influence=1
                        strip.frame_start=0
                        strip.extrapolation='NOTHING'
        assert found==set(case.get('layers',[])),(found,case)
    bpy.context.scene.frame_set(-1)
    frame=case['t']*24
    bpy.context.scene.frame_set(int(frame),subframe=frame-int(frame))
    depsgraph=bpy.context.evaluated_depsgraph_get()
    points=[]
    for obj in bpy.context.scene.objects:
        if obj.type!='MESH' or any(c.hide_render for c in obj.users_collection): continue
        evaluated=obj.evaluated_get(depsgraph);mesh=evaluated.to_mesh()
        p=np.empty(len(mesh.vertices)*3,dtype=np.float32)
        mesh.vertices.foreach_get('co',p)
        matrix=np.asarray(evaluated.matrix_world)
        points.append(p.reshape(-1,3)@matrix[:3,:3].T+matrix[:3,3])
        evaluated.to_mesh_clear()
    file=request.parent/f'blender-{index:02d}.npy'
    np.save(file,np.concatenate(points))
    results.append({**case,'points':file.name})
request.with_suffix('.result.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print('CHARACTER_BLENDER_PLAYBACK_OK')
