"""Post-rig preparation: a model proposes JSON; a frozen job owns limits and delivery.

No shell, script execution or model-supplied filesystem paths. Offline seam: run(root, chat_fn).
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import time
from pathlib import Path

from . import agent, game, game_check, game_scene_review, game_measure, game_package, game_runtime, game_scenario, llm, terrain

CRITERIA = dict(loop_seam_mm_max=2, contact_gap_mm_max=5, stance_s_min=.05, root_travel_mm_max=5,
                skate_mps_max=.35, gait_phase_difference_max=.1, blend_slip_mps_max=.35,
                fade_joint_speed_mps_max=8)
SCENARIO_CRITERIA = dict(penetration_mm_max=8, stretch_mm_max=1, joint_speed_mps_max=15,
                         contact_gap_mm_max=20, slip_mps_max=.35, support_fraction_min=.8,
                         limit_violation_deg_max=1, playback_mm_max=.5)
MOTIONS = {'measured', 'authored', 'stationary', 'airborne'}
SYSTEM = """Prepare ONE frozen animated character for runtime blending and ground IK. You write JSON only.
The job owns the required legs/gaits, criteria and scenarios. You cannot change the rig or weaken its checks.
Use the supplied game-info measurements: name actual joints, bound the sole contact regions, read bend ranges,
and choose poles in Blender axes (+Z up, +Y forward). Describe every required leg and every existing clip.
Call build with complete character_json and playback_json strings; read every failure before revising. You may use
clip, states and blend1d graphs, with two_bone ground legs. Declare every required gait as a blend1d point
(its value may be measured). Add action clips via state layers. Omit criteria: the harness inserts its limits.
Use positive fade times. Ground adaptation needs body, drop/rise limits, tilt (only with >=3 legs), relock_m,
blend_s and lookahead_s. All clips must be represented in the graph, even non-locomotion ones.
Playback is {version:1, cycle:[{clip, motion, cycles, exit, reference?}, ...]}, each clip exactly once.
motion is measured for gaits (their measured forward speed), authored for horizontal root travel, stationary
for idle/actions that stay put, or airborne for an in-place jump travelling at a required gait's speed only
while all feet are airborne (name that gait as reference). cycles is 1..4; exit is end or landing. Landing
requires a fully airborne interval. Do not rename clips, invent motion or omit difficult feet. The full
individual clip will repeat when selected in the viewer. Build also returns two-angle temporal scene pictures,
including an IK-off source control and obstacle stress. Inspect the actual skinned feet and body across adjacent
frames: stance labels distinguish planted contacts from normal swing/flight. Look for penetration, sliding,
target flips, unnatural folds and body jumps; a smaller gap alone is not improvement. Call review with specific
clip/time/limb observations and needs_revision before revising or delivering. You cannot review in the same
turn as build: its pictures arrive afterward. Scene stress is diagnostic; it cannot waive numeric failures.
Call done with the best numerically passing and visually reviewed attempt. If nothing
passes, call report_failure and explain the frozen rig/clip defects. A valid contract is not evidence of good animation or physics."""


def _fn(name, description, properties, required):
    return {'type': 'function', 'function': {'name': name, 'description': description,
            'parameters': {'type': 'object', 'properties': properties, 'required': required}}}


CONTRACT_GUIDE = """CHARACTER JSON SHAPE (replace example names using game-info; include every required gait and clip):
{\"version\":1,\"rig_sha256\":\"<provided hash>\",\"parameters\":{\"speed\":{\"min\":0,\"max\":10,\"default\":0}},
 \"graph\":{\"layers\":[{\"name\":\"gait\",\"node\":{\"type\":\"blend1d\",\"parameter\":\"speed\",\"sync\":false,
 \"points\":[{\"clip\":\"idle\",\"value\":0},{\"clip\":\"walk\",\"value\":\"measured\"},{\"clip\":\"run\",\"value\":\"measured\"}]}},
 {\"name\":\"actions\",\"node\":{\"type\":\"states\",\"start\":\"off\",\"states\":{\"off\":null,\"jump\":{\"type\":\"clip\",\"clip\":\"jump\"}},
 \"transitions\":[{\"from\":\"off\",\"to\":\"jump\",\"on\":\"jump\",\"fade_s\":0.2},{\"from\":\"jump\",\"to\":\"off\",\"at_end\":true,\"fade_s\":0.2}]}}]},
 \"body\":{\"joint\":\"hips\"},\"limbs\":[{\"name\":\"leg.L\",\"role\":\"leg\",\"joints\":[\"hip.L\",\"knee.L\",\"foot.L\"],
 \"solver\":\"two_bone\",\"pole\":[0,1,0],\"bend_degrees\":[1,179.9],\"contact\":{\"node\":\"sole.L\",\"box\":[[0,0,0],[1,1,1]]}}],
 \"ground\":{\"drop_m_max\":0.3,\"rise_m_max\":0.3,\"tilt\":false,\"relock_m\":0.15,\"blend_s\":0.08,\"lookahead_s\":0.2}}
This is a syntax example, not a candidate: use actual clips, body, chains, poles and contact boxes. Add every leg
and action state. A synced blend additionally needs compatible footfall phases; sync:false is a valid explicit
choice for incompatible frozen gaits. The package builder resolves measured point values. Criteria are supplied
by the harness. Contact boxes are in Blender rest coordinates; pick the sole vertices, not an arbitrary box.
"""
TOOLS = [
    _fn('build', 'Measure/package this character JSON and check every job scenario. Each call uses an attempt.',
        {'character_json': {'type': 'string'}, 'playback_json': {'type': 'string'}}, ['character_json', 'playback_json']),
    _fn('review', 'Record your inspection of the scene pictures already shown for this attempt. Cite clip/time/limb and explain any revision.',
        {'attempt': {'type':'string'}, 'observations': {'type':'string'}, 'needs_revision': {'type':'boolean'}},
        ['attempt','observations','needs_revision']),
    _fn('done', 'Deliver a passing attempt (a01 etc.), or report why the frozen rig cannot pass.',
        {'best': {'type': 'string'}, 'summary': {'type': 'string'}}, ['best', 'summary']),
    _fn('report_failure', 'Stop without delivery when the frozen rig/clip defects prevent acceptance.',
        {'summary': {'type':'string'}}, ['summary']),
]


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(game_package.dump(value))


def init(rig, out, *, legs, gaits):
    """Create a new, owner-editable acceptance job. The agent never writes these inputs."""
    rig, out = Path(rig).resolve(), Path(out).resolve()
    if out.exists():
        raise ValueError('the job directory must be new')
    if type(legs) is not int or not 1 <= legs <= 32:
        raise ValueError('legs must be 1..32')
    body = game_measure.Body(rig)
    if not gaits or not set(gaits) <= set(body.clips) or len(set(gaits)) != len(gaits):
        raise ValueError('gaits must name distinct existing clips')
    job = {'version': 1, 'rig_sha256': terrain.digest(rig), 'legs': legs, 'gaits': list(gaits),
           'clips': list(body.clips), 'criteria': dict(CRITERIA), 'scenario_criteria': dict(SCENARIO_CRITERIA)}
    out.mkdir(parents=True)
    shutil.copyfile(rig, out/'animated.glb')
    dump(out/'game.json', job)
    (out/'brief.md').write_text('Prepare ground adaptation and all-clip playback. Keep the input GLB unchanged.\n', encoding='utf-8')
    return job


def load(root):
    root = Path(root).resolve()
    for name in ('game.json', 'animated.glb', 'brief.md'):
        if not (root/name).is_file() or (root/name).is_symlink() or not (root/name).resolve().is_relative_to(root):
            raise ValueError(f'missing or unsafe game input: {name}')
    job = json.loads((root/'game.json').read_text(encoding='utf-8'))
    if set(job) != {'version','rig_sha256','legs','gaits','clips','criteria','scenario_criteria'} or job['version'] != 1:
        raise ValueError('invalid game job fields/version')
    if type(job['legs']) is not int or not 1 <= job['legs'] <= 32:
        raise ValueError('invalid required leg count')
    if terrain.digest(root/'animated.glb') != job['rig_sha256']:
        raise ValueError('the frozen rig hash changed')
    body = game_measure.Body(root/'animated.glb')
    if (set(job['clips']) != set(body.clips) or len(job['clips']) != len(body.clips)
            or not job['gaits'] or not set(job['gaits']) <= set(job['clips'])):
        raise ValueError('job clips/gaits do not match the rig')
    if len(body.clips) > 32 or any(c.duration <= 0 or c.duration > 30 for c in body.clips.values()):
        raise ValueError('game agent supports at most 32 clips of up to 30 seconds')
    if set(job['criteria']) != set(CRITERIA) or set(job['scenario_criteria']) != set(SCENARIO_CRITERIA):
        raise ValueError('job must state every acceptance limit')
    for block in ('criteria','scenario_criteria'):
        for key, value in job[block].items():
            terrain.finite(value, block+'.'+key, positive=True)
    return root, job


def playback_policy(value, clips, gaits):
    if not isinstance(value, dict) or set(value) != {'version','cycle'} or value['version'] != 1:
        raise ValueError('invalid playback policy')
    cycle = value['cycle']
    if not isinstance(cycle, list) or len(cycle) != len(clips):
        raise ValueError('playback must include every clip exactly once')
    names = []
    for row in cycle:
        if not isinstance(row, dict) or set(row)-{'clip','motion','cycles','exit','reference'}:
            raise ValueError('invalid playback entry fields')
        if not {'clip','motion','cycles','exit'} <= set(row):
            raise ValueError('playback entry needs clip, motion, cycles and exit')
        if row['clip'] not in clips or row['motion'] not in MOTIONS or row['exit'] not in ('end','landing'):
            raise ValueError('invalid playback clip/motion/exit')
        if type(row['cycles']) is not int or not 1 <= row['cycles'] <= 4:
            raise ValueError('playback cycles must be 1..4')
        if row['motion'] == 'airborne':
            if row.get('reference') not in gaits:
                raise ValueError('airborne motion needs a declared gait reference')
        elif 'reference' in row:
            raise ValueError('only airborne motion uses a reference')
        if row['clip'] in gaits and row['motion'] != 'measured':
            raise ValueError('required gaits must consume measured travel')
        names.append(row['clip'])
    if len(set(names)) != len(names) or set(names) != set(clips):
        raise ValueError('playback clips must be unique and complete')
    return copy.deepcopy(value)


def contract(proposal, job):
    if not isinstance(proposal,dict):raise ValueError('character proposal must be an object')
    spec = copy.deepcopy(proposal)
    if len(json.dumps(spec)) > 256_000:
        raise ValueError('character proposal is too large')
    spec.setdefault('version', 1)
    spec.setdefault('rig_sha256', job['rig_sha256'])
    if spec['rig_sha256'] != job['rig_sha256']:
        raise ValueError('the agent cannot replace the rig')
    found = game.graph(spec)
    limits = {k: job['criteria'][k] for k in game.needed_criteria(spec, found)}
    if 'criteria' in spec and spec['criteria'] != limits:
        raise ValueError('the agent cannot change acceptance limits')
    spec['criteria'] = limits
    game.contract(spec)
    def supported(n):
        if n is None:return
        if n['type'] not in ('clip','states','blend1d'):raise ValueError('unsupported browser graph node')
        if n['type']=='states':
            for v in n['states'].values():supported(v)
        if n['type']=='blend1d':
            for p in n['points']:
                if 'node' in p:supported(p['node'])
    for layer in spec['graph']['layers']:supported(layer['node'])
    if not spec.get('ground') or len(spec.get('limbs', [])) != job['legs']:
        raise ValueError('every required ground leg must be present')
    if any(l['solver'] != 'two_bone' or l['role'] != 'leg' for l in spec['limbs']) or spec.get('aims'):
        raise ValueError('this browser profile requires two-bone ground legs without aims')
    if set(found['clips']) != set(job['clips']) or not set(job['gaits']) <= set(found['locomotion']):
        raise ValueError('graph must include all clips and declare the required gaits as locomotion')
    return spec


def flight_window(clip):
    """First measured flight followed by at least half the feet landing (also wrapped stances)."""
    legs = list(clip['legs'].values())
    edges = sorted({0., 1., *(v-shift for l in legs for pair in l['stance_phase']
                             for v in pair for shift in (0,1) if 0 < v-shift < 1)})
    takeoff = None
    for a,b in zip(edges,edges[1:]):
        phase = (a+b)/2
        count = sum(any(lo <= phase+shift <= hi for lo,hi in l['stance_phase'] for shift in (0,1)) for l in legs)
        if not count and takeoff is None:takeoff = a
        if takeoff is not None and count >= (len(legs)+1)//2:
            return takeoff*clip['duration_s'], a*clip['duration_s']
    return None


def measured_policy(policy, runtime):
    for row in policy['cycle']:
        if (row['exit']=='landing' or row['motion']=='airborne') and not flight_window(runtime['clips'][row['clip']]):
            raise ValueError(row['clip']+': playback needs a measured flight and landing')
        if row['motion']=='airborne' and (runtime['clips'][row['reference']]['travel_speed_mps'] or 0)<=0:
            raise ValueError('airborne reference must have positive measured speed')


def scenarios(runtime, job):
    """Harness-owned slope checks at measured gait speeds; no model-authored scenario escape."""
    for clip in job['gaits']:
        speed = runtime['clips'][clip]['travel_speed_mps']
        if speed is None or speed <= 0:
            raise ValueError(f'{clip}: no positive measured gait speed')
        for slope,ground_enabled in ((0.0,False),(0.0,True),(.08,True)):
            duration = 2.0
            zs = [-speed*duration-2, 2]
            field = {'x': [-2, 0, 2], 'z': zs,
                     'heights': [[slope*x for x in (-2, 0, 2)] for _ in zs]}
            events = []
            offset = 0
            # Any declared speed parameter is driven through its measured gait point.
            for layer in runtime['graph']['layers']:
                n = layer['node']
                if n['type'] == 'blend1d' and any(p.get('clip') == clip for p in n['points']):
                    point = next(p for p in n['points'] if p.get('clip') == clip)
                    events.append({'time': 0, 'type': 'set_parameter', 'name': n['parameter'], 'value': point['value']})
                    offset = point.get('phase_offset',0)*runtime['clips'][clip]['duration_s']
            if not ground_enabled:
                events.append({'time':0,'type':'set_ground_enabled','enabled':False})
            spec = {'version': 1, 'duration': duration, 'dt': [1/60,1/45], 'terrain': field,
                    'root': [[0,0,-runtime['measurement']['rest_ground_y'],0,0],
                             [duration,0,-runtime['measurement']['rest_ground_y'],-speed*duration,0]],
                    'events': events, 'criteria': {}}
            if not ground_enabled:spec['expect']={'clip':clip,'offset_s':offset}
            spec['criteria'] = {k: job['scenario_criteria'][k] for k in game_scenario.needed_criteria(spec,runtime)}
            yield f'{clip}-'+('slope' if slope else 'flat' if ground_enabled else 'playback'), spec


class Session:
    def __init__(self, root, out, builds):
        self.root, self.job = load(root)
        self.out = Path(out).resolve()
        if not self.out.is_relative_to(self.root) or self.out.exists():
            raise ValueError('run output must be a new directory inside the game job')
        if type(builds) is not int or not 1 <= builds <= 20:
            raise ValueError('builds must be 1..20')
        self.frozen = {n: terrain.digest(self.root/n) for n in ('game.json','animated.glb','brief.md')}
        self.out.mkdir(parents=True)
        self.builds, self.results, self.final, self.failure = builds, {}, None, ''
        self.pictures, self.presented = {}, set()

    def check_frozen(self):
        if any((self.root/n).is_symlink() or terrain.digest(self.root/n) != sha for n,sha in self.frozen.items()):
            raise ValueError('a frozen game input changed during the run')

    def build(self, proposal, playback):
        self.check_frozen()
        if len(self.results) >= self.builds:
            raise ValueError('build budget exhausted')
        if any(r.get('scene_review',{}).get('status')=='pending' for r in self.results.values()):
            raise ValueError('review the previous scene pictures before another build')
        name = f'a{len(self.results)+1:02d}'
        target = self.out/'attempts'/name
        result = {'passed': False}
        self.results[name] = result                 # invalid proposals also consume an attempt
        try:
            dump(target/'proposal.json',{'contract':proposal,'playback':playback})
            if isinstance(proposal,str):proposal=json.loads(proposal)
            if isinstance(playback,str):playback=json.loads(playback)
            spec = contract(proposal, self.job)
            policy = playback_policy(playback, self.job['clips'], self.job['gaits'])
            dump(target/'character.json', spec)
            dump(target/'playback.json', policy)
            package = target/'package'
            report = game_package.build(self.root/'animated.glb', spec, package, diagnostic=True)
            result.update(package_passed=report['passed'], issues=report['issues'],
                          gates=report['gates'], readings=report['readings'])
            preview = package if report['passed'] else package/'diagnostic'
            if (preview/'character.runtime.json').is_file():
                runtime = json.loads((preview/'character.runtime.json').read_text())
                policy_ok = True
                try:
                    measured_policy(policy,runtime)
                except ValueError as error:
                    result['issues'].append(str(error)); policy_ok = False
                if len(runtime['skeleton']['roots'])!=1:raise ValueError('browser profile needs one skeleton root')
                checks, cases = {}, []
                for label, scenario in scenarios(runtime, self.job):
                    trace = game_runtime.run(preview, scenario)
                    checked = game_check.check(preview, scenario, trace)
                    dump(target/(label+'.scenario.json'), scenario)
                    dump(target/(label+'.check.json'), checked)
                    checks[label] = {'passed': checked['passed'], 'issues': checked['issues'], 'gates': checked['gates']}
                    cases.append({'label':label,'scenario':scenario,'trace':trace,'check':checked})
                coverage=not report['readings'].get('missed_contacts')
                result.update(checks=checks, passed=report['passed'] and policy_ok and coverage and bool(checks)
                              and all(c['passed'] for c in checks.values()))
                if not coverage:result['issues'].append('declared contacts miss measured ground-contact clusters')
                result['issues'] += [label+': '+issue for label,c in checks.items() if not c['passed'] for issue in c['issues']]
                paths = game_scene_review.capture(preview,cases,target/'scene-review')
                if not paths:raise ValueError('scene review produced no pictures')
                self.pictures[name] = paths
                result['scene_review'] = {'status':'pending', 'pictures':{
                    p.relative_to(self.root).as_posix():terrain.digest(p) for p in paths}}
        except (ValueError, KeyError, TypeError, IndexError, OSError) as error:
            result['passed'] = False
            result.setdefault('issues',[]).append(str(error))
        self.check_frozen()
        dump(target/'result.json', result)
        return {'attempt': name, **result}

    def review(self, attempt, observations, needs_revision):
        self.check_frozen()
        if attempt not in self.presented:
            raise ValueError('review requires scene pictures from an earlier model turn')
        if not isinstance(observations,str) or not observations.strip() or len(observations)>8000:
            raise ValueError('review needs concrete observations, at most 8000 characters')
        if type(needs_revision) is not bool:raise ValueError('needs_revision must be boolean')
        result = self.results[attempt]
        review = result['scene_review']
        if any(terrain.digest(self.root/p)!=sha for p,sha in review['pictures'].items()):
            raise ValueError('scene pictures changed after presentation')
        review.update(status='reviewed',observations=observations.strip(),needs_revision=needs_revision)
        dump(self.out/'attempts'/attempt/'result.json',result)
        return {'attempt':attempt, **review}

    def done(self, best, summary):
        self.check_frozen()
        if not isinstance(best,str) or best not in self.results or not self.results[best]['passed']:
            raise ValueError('delivery needs a passing attempt from this run')
        review = self.results[best].get('scene_review',{})
        if review.get('status')!='reviewed' or review.get('needs_revision',True):
            raise ValueError('delivery needs scene review with no outstanding revision')
        if any(terrain.digest(self.root/p)!=sha for p,sha in review['pictures'].items()):
            raise ValueError('scene pictures changed after review')
        self.final = {'best': best, 'summary': str(summary), 'package': f'attempts/{best}/package'}
        return dict(self.final)


def run(root, chat_fn, *, out, builds=5, max_turns=0, model='', log=print):
    session = Session(root,out,builds)
    messages = [{'role':'system','content':SYSTEM+'\n'+CONTRACT_GUIDE}, {'role':'user','content':json.dumps(session.job)+'\n'+
                (session.root/'brief.md').read_text(encoding='utf-8')+'\n'+game.info(session.root/'animated.glb')}]
    turns = max_turns or builds*4+4
    if type(turns) is not int or not 1<=turns<=200:raise ValueError('max_turns must be 1..200')
    calls = 0
    stop = 'turn limit'
    for _ in range(turns):
        session.check_frozen()
        try:
            message, usage = chat_fn(messages, TOOLS)
        except (llm.LLMError, OSError, TimeoutError):
            stop = 'transport failed'
            session.failure = 'The model call failed; no delivery was accepted.'
            break
        calls += 1
        messages.append(message)
        agent.atomic_write_text(session.out/'transcript.jsonl','\n'.join(json.dumps(m,allow_nan=False) for m in messages)+'\n')
        tool_calls = message.get('tool_calls') or []
        if not tool_calls:
            session.failure=str(message.get('content') or '')
            stop = 'model ended without a passing delivery'
            break
        pending = []
        for call in tool_calls:
            try:
                args = json.loads(call['function']['arguments'])
                if call['function']['name'] == 'build':
                    result = session.build(args.get('character_json'),args.get('playback_json'))
                    if result['attempt'] in session.pictures:pending.append(result['attempt'])
                    log(f"game {result['attempt']}: {'passed' if result['passed'] else 'failed'}")
                    for issue in result.get('issues',[])[:3]:log('  '+issue)
                elif call['function']['name'] == 'review':
                    result = session.review(args['attempt'],args['observations'],args['needs_revision'])
                elif call['function']['name'] == 'done':
                    result = session.done(args['best'],args['summary'])
                elif call['function']['name'] == 'report_failure':
                    session.failure=str(args['summary']);result={'delivered':False,'summary':session.failure}
                else:
                    raise ValueError('unknown game tool')
            except (ValueError,KeyError,TypeError) as error:
                result = {'error':str(error)}
            messages.append({'role':'tool','tool_call_id':call['id'],'content':json.dumps(result,allow_nan=False)})
            if session.final or session.failure:break
        if pending and not (session.final or session.failure):
            parts = []
            for name in pending:
                parts.append({'type':'text','text':f'Scene pictures for {name}. Read stance and time labels; compare source-off, slope and obstacle strips. '
                              'These are simplified CPU reference scenes, not screenshots of the consuming browser/game. '
                              'Review concrete defects before another build or done. Numeric failures remain failures.'})
                for path in session.pictures[name]:
                    parts.append({'type':'image_url','image_url':{'url':llm.image_to_data_url(str(path),max_dim=2048)}})
                session.presented.add(name)
            messages.append({'role':'user','content':parts})
        agent.atomic_write_text(session.out/'transcript.jsonl','\n'.join(json.dumps(m,allow_nan=False) for m in messages)+'\n')
        if session.final:
            stop = 'delivered'
            break
        if session.failure:
            stop='failed checks';break
    report = {'version':1,'model':model,'calls':calls,'stop':stop,'summary':session.failure,'final':session.final,
              'attempts':session.results,'inputs':session.frozen}
    dump(session.out/'report.json', report)
    return report


def drive(root, backend, *, builds=5, max_turns=0, max_tokens=8192, max_cost=0, effort=None, idle_s=900, log=print):
    from . import anthropic_sweep, gauntlet
    from .gauntlet_sweep import Meter, estimate_cost
    root = Path(root).resolve()
    native = backend.url == gauntlet.URL and backend.model in anthropic_sweep.RATES
    if max_cost and not native:
        raise ValueError('reserved spend caps require the direct Anthropic transport')
    out = root/'agent'/(time.strftime('%Y%m%d-%H%M%S')+'_'+re.sub(r'[^A-Za-z0-9._-]+','-',backend.model)+'_game')
    budget = anthropic_sweep.Budget(out/'budget.json',max_cost,backend.model) if max_cost else None
    if native:
        chat = anthropic_sweep.chat_for(backend,budget,'game-agent',max_tokens=max_tokens or 8192,effort=effort,log=log,count_tokens=bool(budget))
    else:
        chat = agent.chat_for(backend,effort=effort,max_tokens=max_tokens or 8192,idle_s=idle_s,log=log)
    # Meter writes files only after Session has safely created the unique run directory.
    meter = Meter(chat,out,backend.model,price_fn=estimate_cost if native else lambda usage,model:None)
    report = run(root,meter,out=out,builds=builds,max_turns=max_turns,model=backend.model,log=log)
    report.update(effort=effort,max_cost=max_cost,limits={'builds':builds,'turns':max_turns or builds*4+4,'reply_tokens':max_tokens or 8192})
    report['usage'] = meter.summary()
    if budget:budget.save()
    dump(out/'report.json',report)
    return report,out


def init_command(args):
    init(args.rig,args.out,legs=args.legs,gaits=args.gaits)
    print(f'Created game job: {args.out}')
    return 0


def add_commands(subs):
    p = subs.add_parser('game-init',help='create a frozen post-rig game-agent job with explicit acceptance limits')
    p.add_argument('--rig',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--legs',type=int,required=True)
    p.add_argument('--gaits',nargs='+',required=True,help='existing gait clip names, as listed by game-info')
    p.set_defaults(fn=init_command)
