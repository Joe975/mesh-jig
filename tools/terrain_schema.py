"""Bounded public browser packages. Authoring contracts and free-form provenance are never accepted."""
from __future__ import annotations
import json
import math
import re


def require(ok, message):
    if not ok:raise ValueError(message)


def fields(value, required, optional=()):
    require(isinstance(value,dict) and set(required)<=set(value)<=set(required)|set(optional),'unknown or missing terrain fields')


def number(value, lo=-1e6, hi=1e6):
    require(type(value) in (int,float) and math.isfinite(value) and lo<=value<=hi,'invalid terrain number')


def vector(value, size=3):
    require(isinstance(value,list) and len(value)==size,'invalid terrain vector')
    for x in value:number(x)


def identifier(value):
    require(isinstance(value,str) and re.fullmatch(r'[A-Za-z][A-Za-z0-9_ .-]{0,63}',value),'invalid public identifier')


def playback(policy, clips, limbs):
    fields(policy,{'version','cycle','camera'})
    require(policy['version']==1 and isinstance(policy['cycle'],list) and len(policy['cycle'])==len(clips),'invalid demo policy')
    names=[]
    for row in policy['cycle']:
        fields(row,{'clip','motion','cycles','exit'},{'reference'})
        require(row['clip'] in clips and row['motion'] in {'measured','authored','stationary','airborne'} and row['exit'] in {'end','landing'},'invalid demo motion')
        require(type(row['cycles']) is int and 1<=row['cycles']<=4,'invalid demo repetitions')
        if row['motion']=='airborne':require(row.get('reference') in clips and clips[row['reference']]['locomotion'] and (clips[row['reference']]['travel_speed_mps'] or 0)>0,'invalid demo gait reference')
        else:require('reference' not in row,'unexpected gait reference')
        if row['motion']=='measured':require((clips[row['clip']]['travel_speed_mps'] or 0)>0,'missing measured demo speed')
        if row['motion']=='airborne' or row['exit']=='landing':
            legs=list(clips[row['clip']]['legs'].values());edges=sorted({0.,1.,*(v-shift for l in legs for pair in l['stance_phase'] for v in pair for shift in (0,1) if 0<v-shift<1)})
            flight=False;landed=False
            for a,b in zip(edges,edges[1:]):
                phase=(a+b)/2;count=sum(any(lo<=phase+shift<=hi for lo,hi in l['stance_phase'] for shift in (0,1)) for l in legs)
                if not count:flight=True
                if flight and count>=(len(legs)+1)//2:landed=True;break
            require(landed,'missing measured demo flight/landing')
        names.append(row['clip'])
    require(len(set(names))==len(names) and set(names)==set(clips),'incomplete demo clips')
    fields(policy['camera'],{'position','target'})
    vector(policy['camera']['position']);vector(policy['camera']['target'])


def payload(raw):
    require(len(raw)<=5_000_000,'terrain payload too large')
    d=json.loads(raw)
    fields(d,{'version','glb_sha256','nodes','runtime','tracks','observations','demo'})
    require(type(d['version']) is int and d['version']==2,'unsupported terrain version')
    require(isinstance(d['glb_sha256'],str) and re.fullmatch('[a-f0-9]{64}',d['glb_sha256']),'invalid character hash')
    nodes=d['nodes'];require(isinstance(nodes,list) and 1<=len(nodes)<=512,'invalid node count')
    parents={}
    for i,node in enumerate(nodes):
        fields(node,set(),{'children','matrix','translation','rotation','scale'})
        for k,size in [('matrix',16),('translation',3),('rotation',4),('scale',3)]:
            if k in node:vector(node[k],size)
        children=node.get('children',[]);require(isinstance(children,list) and len(set(children))==len(children),'invalid node children')
        for c in children:
            require(type(c) is int and 0<=c<len(nodes) and c not in parents,'invalid node hierarchy');parents[c]=i
    for i in range(len(nodes)):
        seen=set();j=i
        while j in parents:
            require(j not in seen,'cyclic node hierarchy');seen.add(j);j=parents[j]
    r=d['runtime'];fields(r,{'graph','parameters','body','ground','measurement','skeleton','limbs','clips','aims'})
    fields(r['skeleton'],{'joints'});joints=r['skeleton']['joints'];require(isinstance(joints,list) and 1<=len(joints)<=256,'invalid skeleton size')
    names={j['name'] for j in joints};require(len(names)==len(joints),'duplicate joints')
    for j in joints:
        fields(j,{'name','node','parent','rest_position'});require(type(j['node']) is int and 0<=j['node']<len(nodes) and j['name']=='j'+str(j['node']) and j['parent'] in names|{None},'invalid anonymous joint');vector(j['rest_position'])
        require('matrix' not in nodes[j['node']],'joint matrix transforms are unsupported')
    by_node={j['node']:j['name'] for j in joints}
    require(sum(j['parent'] is None for j in joints)==1,'terrain needs one skeleton root')
    for j in joints:
        ancestor=parents.get(j['node'])
        while ancestor is not None and ancestor not in by_node:ancestor=parents.get(ancestor)
        require(j['parent']==by_node.get(ancestor),'joint parent differs from node hierarchy')
    fields(r['body'],{'joint'});require(r['body']['joint'] in names,'invalid body joint')
    require(r['aims']==[],'unsupported aims')
    params=r['parameters'];require(isinstance(params,dict) and len(params)<=16,'invalid parameters')
    for k,p in params.items():
        require(re.fullmatch('p[0-9]+',k),'nonanonymous parameter');fields(p,{'min','max','default'})
        for v in p.values():number(v)
        require(p['min']<p['max'] and p['min']<=p['default']<=p['max'],'invalid parameter range')
    clips=r['clips'];require(isinstance(clips,dict) and 1<=len(clips)<=32,'invalid clip count')
    legs=r['limbs'];require(isinstance(legs,list) and 1<=len(legs)<=32,'invalid leg count')
    leg_names={l['name'] for l in legs};require(len(leg_names)==len(legs),'duplicate legs')
    for l in legs:
        fields(l,{'name','role','joints','solver','pole','bend_degrees','segment_lengths_m','contact','clearance'})
        require(re.fullmatch('leg[0-9]+',l['name']) and l['role']=='leg' and l['solver']=='two_bone' and len(l['joints'])==3 and len(set(l['joints']))==3 and all(n in names for n in l['joints']),'unsupported terrain limb')
        vector(l['pole']);vector(l['bend_degrees'],2);vector(l['segment_lengths_m'],2)
        require(all(v>0 for v in l['segment_lengths_m']) and 0<=l['bend_degrees'][0]<l['bend_degrees'][1]<=180,'invalid limb geometry')
        for k,lo,hi in [('contact',1,2048),('clearance',0,32)]:
            fields(l[k],{'points'});points=l[k]['points'];require(isinstance(points,list) and lo<=len(points)<=hi,'invalid foot probes')
            for p in points:vector(p)
    for name,c in clips.items():
        identifier(name);fields(c,{'duration_s','loop','travel_speed_mps','legs','locomotion'})
        number(c['duration_s'],1e-6,30);require(type(c['loop']) is bool and type(c['locomotion']) is bool,'invalid clip flags')
        if c['travel_speed_mps'] is not None:number(c['travel_speed_mps'],-100,100)
        require(set(c['legs'])==leg_names,'incomplete stance observations')
        for v in c['legs'].values():
            fields(v,{'duty','speed_mps','spread_mps','stance_phase','touchdown_phase'});number(v['duty'],0,1)
            for key in ('speed_mps','spread_mps'):
                if v[key] is not None:number(v[key],-100,100)
            require(isinstance(v['stance_phase'],list) and len(v['stance_phase'])<=128,'invalid stance windows')
            for a,b in v['stance_phase']:number(a,0,2);number(b,a,2)
            require(isinstance(v['touchdown_phase'],list) and len(v['touchdown_phase'])<=128,'invalid touchdowns')
            for x in v['touchdown_phase']:number(x,0,2)
    def graph(n,depth=0):
        require(depth<=12 and isinstance(n,dict),'invalid graph nesting')
        kind=n.get('type')
        if kind=='clip':fields(n,{'type','clip'});require(n['clip'] in clips,'unknown graph clip')
        elif kind=='states':
            fields(n,{'type','start','states','transitions'});states=n['states'];require(isinstance(states,dict) and 1<=len(states)<=64 and n['start'] in states,'invalid state graph')
            for key,v in states.items():
                require(re.fullmatch('s[0-9]+',key),'nonanonymous state')
                if v is not None:graph(v,depth+1)
            require(isinstance(n['transitions'],list) and len(n['transitions'])<=256,'invalid transitions')
            for t in n['transitions']:
                fields(t,{'from','to','fade_s'},{'on','at_end'});require(t['from'] in states and t['to'] in states and ('on' in t)!=('at_end' in t),'invalid transition')
                number(t['fade_s'],1e-6,30)
                if 'on' in t:require(isinstance(t['on'],str) and re.fullmatch('t[0-9]+',t['on']),'nonanonymous trigger')
                else:require(t['at_end'] is True,'invalid end transition')
        elif kind=='blend1d':
            fields(n,{'type','parameter','points'},{'sync','sync_clock'});require(n['parameter'] in params,'unknown graph parameter')
            points=n['points'];require(isinstance(points,list) and 2<=len(points)<=32,'invalid blend points')
            if 'sync' in n:require(type(n['sync']) is bool,'invalid sync flag')
            if 'sync_clock' in n:require(n['sync_clock']=='frequency_lerp','unsupported blend clock')
            values=[]
            for p in points:
                fields(p,{'value'},{'clip','node','phase_offset','cycle','measured'});number(p['value']);values.append(p['value'])
                require(('clip' in p)!=('node' in p),'invalid blend point')
                if 'clip' in p:require(p['clip'] in clips,'unknown blend clip')
                else:graph(p['node'],depth+1)
                if 'phase_offset' in p:number(p['phase_offset'],0,1)
                for k in ('cycle','measured'):
                    if k in p:require(type(p[k]) is bool,'invalid measured flag')
                if n.get('sync'):require('clip' in p and 'cycle' in p and 'phase_offset' in p,'incomplete synced point')
            require(all(a<b for a,b in zip(values,values[1:])),'unordered blend points')
        else:raise ValueError('unsupported browser graph node')
    fields(r['graph'],{'layers'});require(isinstance(r['graph']['layers'],list) and 1<=len(r['graph']['layers'])<=16,'invalid graph layers')
    for layer in r['graph']['layers']:
        fields(layer,{'node','mask_joints'});require(layer['mask_joints'] is None or isinstance(layer['mask_joints'],list) and all(n in names for n in layer['mask_joints']),'invalid joint mask');graph(layer['node'])
    fields(r['ground'],{'drop_m_max','rise_m_max','tilt','relock_m','blend_s','lookahead_s'})
    for k,v in r['ground'].items():
        if k=='tilt':require(type(v) is bool,'invalid tilt flag')
        else:number(v,0,10)
    fields(r['measurement'],{'sample_hz','contact_gap_m','rest_ground_y'})
    for v in r['measurement'].values():number(v)
    require(set(d['tracks'])==set(clips),'incomplete clip tracks')
    for name,channels in d['tracks'].items():
        require(isinstance(channels,list) and 1<=len(channels)<=2048,'invalid track count')
        for ch in channels:
            fields(ch,{'node','path','interpolation','times','values'})
            require(type(ch['node']) is int and 0<=ch['node']<len(nodes) and ch['path'] in {'translation','rotation','scale'} and ch['interpolation'] in {'LINEAR','STEP'},'unsupported track')
            require('matrix' not in nodes[ch['node']],'animated matrix transforms are unsupported')
            times=ch['times'];require(isinstance(times,list) and 1<=len(times)<=4096 and all(a<b for a,b in zip(times,times[1:])),'invalid track times')
            for t in times:number(t,0,clips[name]['duration_s']+1e-4)
            values=ch['values'];require(isinstance(values,list) and len(values)==len(times)*(4 if ch['path']=='rotation' else 3),'invalid track values')
            for v in values:number(v)
    require(set(d['observations'])==leg_names,'incomplete observed feet')
    for ids in d['observations'].values():
        require(isinstance(ids,list) and 1<=len(ids)<=8192,'invalid observed contacts')
        for triple in ids:require(isinstance(triple,list) and len(triple)==3 and all(type(i) is int and 0<=i<10_000_000 for i in triple) and triple[0]<len(nodes),'invalid contact index')
    playback(d['demo'],clips,legs)
    return d


def registry(raw):
    require(len(raw)<=64_000,'registry too large');d=json.loads(raw);fields(d,{'version','characters'})
    require(d['version']==1 and isinstance(d['characters'],dict) and 1<=len(d['characters'])<=64,'invalid character registry')
    for key,row in d['characters'].items():
        require(re.fullmatch('[a-z][a-z0-9-]{0,47}',key),'invalid character key')
        fields(row,{'data','model','sha256','glb_sha256'})
        require(re.fullmatch('[a-z][a-z0-9-]{0,47}\\.json',row['data']),'unsafe character data path')
        require(re.fullmatch('showcase/[a-z0-9-]+/model-[0-9]+\\.glb',row['model']),'unsafe character model path')
        for k in ('sha256','glb_sha256'):require(re.fullmatch('[a-f0-9]{64}',row[k]),'invalid registry hash')
    return d


def legacy_payload(data):
    """Bounded numeric runtime schema, with an explicit vocabulary for this pilot."""
    if len(data)>1_000_000:
        raise ValueError('terrain payload too large')
    d=json.loads(data)
    if set(d)!={'version','glb_sha256','nodes','runtime','tracks','observations'} or d['version']!=1:
        raise ValueError('unknown terrain payload fields')
    keys=set('version glb_sha256 nodes runtime tracks observations children matrix translation rotation scale graph layers mask_joints name node type clip start states transitions from to on at_end fade_s body joint ground blend_s drop_m_max lookahead_s relock_m rise_m_max tilt measurement contact_gap_m rest_ground_y sample_hz skeleton joints parent rest_position limbs role solver pole bend_degrees segment_lengths_m contact clearance points clips duration_s loop travel_speed_mps legs duty speed_mps spread_mps stance_phase touchdown_phase locomotion aims path interpolation times values sprint leap off base'.split())
    words={'LINEAR','STEP','translation','rotation','scale','clip','states','base','leap','off','sprint','two_bone','leg'}
    def walk(x,depth=0):
        if depth>16:raise ValueError('terrain nesting')
        if isinstance(x,dict):
            for k,v in x.items():
                if k not in keys and not re.fullmatch(r'j\d+|leg\d+',k):raise ValueError('unknown terrain key')
                walk(v,depth+1)
        elif isinstance(x,list):
            if len(x)>8192:raise ValueError('terrain array too large')
            for v in x:walk(v,depth+1)
        elif isinstance(x,str):
            if x not in words and not re.fullmatch(r'j\d+|leg\d+|[a-f0-9]{64}',x):raise ValueError('unknown terrain string')
        elif isinstance(x,(int,float)) and not isinstance(x,bool):
            if not math.isfinite(x) or abs(x)>1e6:raise ValueError('invalid terrain number')
        elif x is not None and not isinstance(x,bool):raise ValueError('invalid terrain value')
    walk(d)
    if not 1<=len(d['nodes'])<=256 or set(d['tracks'])!={'sprint','leap'}:
        raise ValueError('unsupported terrain capabilities')
    r=d['runtime']
    if set(r)!={'graph','body','ground','measurement','skeleton','limbs','clips','aims'} or r['aims'] or len(r['limbs'])!=6:
        raise ValueError('unsupported terrain runtime')
    names={j['name'] for j in r['skeleton']['joints']}
    for j in r['skeleton']['joints']:
        if j['name']!='j'+str(j['node']) or not 0<=j['node']<len(d['nodes']) or j['parent'] not in names|{None}:
            raise ValueError('invalid terrain joint')
    for leg in r['limbs']:
        if leg['solver']!='two_bone' or len(leg['joints'])!=3 or any(n not in names for n in leg['joints']):raise ValueError('invalid terrain limb')
        if not 1<=len(leg['contact']['points'])<=256 or len(leg['clearance']['points'])>32:raise ValueError('invalid terrain samples')
    for channels in d['tracks'].values():
        if not 1<=len(channels)<=256:raise ValueError('invalid terrain channels')
        for ch in channels:
            dim=4 if ch['path']=='rotation' else 3
            if not 0<=ch['node']<len(d['nodes']) or len(ch['values'])!=dim*len(ch['times']) or not ch['times'] or any(b<=a for a,b in zip(ch['times'],ch['times'][1:])):raise ValueError('invalid terrain keys')
    return d
