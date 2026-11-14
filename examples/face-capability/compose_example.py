"""Offline body/face composition, with independently playable and baked clips.

python examples/face-capability/compose_example.py local-work/combined-puppet
"""
import argparse
import json
from pathlib import Path

from mesh_jig import character, rig
from make_example import binding, write_source

BODY_SCRIPT = '''import math

def skeleton(model):
    return [dict(name='root',head=[0,0,-1.8],tail=[0,0,-1.6],parent=None),
            dict(name='body',head=[0,0,-.9],tail=[0,0,-.4],parent='root'),
            dict(name='head',head=[0,0,0],tail=[0,0,.3],parent='body'),
            dict(name='leg_l',head=[-.2,0,-1],tail=[-.2,0,-1.8],parent='root'),
            dict(name='leg_r',head=[.2,0,-1],tail=[.2,0,-1.8],parent='root')]

def weights(name,position,piece):
    return name if name in ('body','leg_l','leg_r') else 'head'

def pose(clip,t):
    wave=math.sin(math.pi*t)
    return {'root':{'move':[.1*wave,.08*t,.035*abs(wave)],'rotate':[('Z',12*wave)]},
            'body':{'rotate':[('X',3*wave)]},'head':{'rotate':[('Z',8*wave),('X',5*wave)]},
            'leg_l':{'rotate':[('X',24*wave)]},'leg_r':{'rotate':[('X',-24*wave)]}}
'''

BODY_CONTRACT={'fps':24,'max_influences':4,'clips':{'walk':{'seconds':2,'loop':False}}}


def make(root):
    root=Path(root)
    root.mkdir(parents=True,exist_ok=False)
    source=root/'source.glb'
    write_source(source,with_body=True)
    script=root/'body.py';script.write_text(BODY_SCRIPT,encoding='utf-8')
    body=root/'body.glb';rig.build(source,script,BODY_CONTRACT,body)
    face=binding()['rig']
    config={'version':1,'anchor':'head','neck':'additive','fps':24,
            'demos':[{'name':'walk_talk','body':'walk','expression':'talk'},
                     {'name':'walk_yell_gaze','body':'walk','expression':'yell','gaze_target':[.5,4,.7]}]}
    (root/'face.json').write_text(json.dumps(face,indent=2),encoding='utf-8')
    (root/'compose.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    output=root/'character.glb'
    character.compose(source,body,face,config,output)
    character.preview(output,root/'talk','walk','talk',seconds=2,fps=12)
    character.preview(output,root/'yell-gaze','walk','yell',[.5,4,.7],seconds=2,fps=12)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    print(make(parser.parse_args().root))
