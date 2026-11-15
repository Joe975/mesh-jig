// Rendered surfaces and ground queries share the same triangles, including obstacles.
const COURSE=[
  {kind:'block',x:-.58,z:1.8,width:.9,length:.85,scale:.85},
  {kind:'rock',x:.65,z:4.1,width:1.15,length:1.25,scale:1},
  {kind:'block',x:-.63,z:6.25,width:1.05,length:.7,scale:.65},
];
// Symmetric so travel in either direction ascends, crosses the landing, then descends.
const STAIRS=[1,2,3,4,3,2,1].map((level,i)=>({kind:'stair',x:0,z:i<3?1.55+(i+.5)*.55:i===3?4:4.8+(i-3-.5)*.55,width:2.8,length:i===3?1.6:.55,level}));
// The model travels toward negative Z: it meets the high-Z staircase first.
const SEQUENCE=[...STAIRS.map(o=>({...o,z:o.z+16})),
  {kind:'rock',x:-.6,z:14,width:1.05,length:1.1,scale:.8},
  {kind:'rock',x:.6,z:12,width:1.15,length:1.25,scale:1},
  {kind:'rock',x:-.15,z:10,width:1.25,length:1,scale:.65},
  {kind:'wall',x:0,z:5.2,width:2.8,length:.45,height:.5},
];
function surfaceHeight(triangle,x,z) {
  const [a,b,c]=triangle,dx=b[0]-a[0],dz=b[2]-a[2],ex=c[0]-a[0],ez=c[2]-a[2],det=dx*ez-dz*ex;
  if(Math.abs(det)<1e-12)return -Infinity; // Vertical walls have no walkable area.
  const u=((x-a[0])*ez-(z-a[2])*ex)/det,v=(dx*(z-a[2])-dz*(x-a[0]))/det;
  return u>=-1e-9&&v>=-1e-9&&u+v<=1+1e-9?a[1]+u*(b[1]-a[1])+v*(c[1]-a[1]):-Infinity;
}

export class Ground {
  constructor(height=.12,obstacleHeight=.24) { this.length=8;this.width=4;this.nx=8;this.nz=32;this.obstacleHeight=obstacleHeight;this.stepHeight=.1;this.course='course';this.obstaclesEnabled=true;this.setHeight(height); }
  setHeight(height) {
    if(!Number.isFinite(height)||height<0||height>.2)throw new Error('Terrain height must be 0..0.2 m');
    this.amplitude=height;this.rows=Array.from({length:this.nz+1},(_,j)=>Array.from({length:this.nx+1},(_,i)=>{
      if(j<=1||j>=this.nz-1)return 0;
      const x=-2+i*4/this.nx,a=j/this.nz*2*Math.PI;
      return height*(.5*(1-Math.cos(a))*(.65+.35*Math.cos(x*1.5))+.25*Math.sin(2*a)*Math.sin(x));
    }));
    this.buildObstacles();
  }
  baseHeight(x,z) {
    const zz=this.wrap(z),xx=Math.max(-2,Math.min(2,x)),fi=(xx+2)/4*this.nx,fj=zz/this.length*this.nz;
    const i=Math.min(this.nx-1,Math.floor(fi)),j=Math.min(this.nz-1,Math.floor(fj)),u=fi-i,v=fj-j;
    const a=this.rows[j][i],b=this.rows[j][i+1],c=this.rows[j+1][i+1],d=this.rows[j+1][i];
    return u>=v?a*(1-u)+b*(u-v)+c*v:a*(1-v)+c*u+d*(v-u);
  }
  setObstacleHeight(height){
    if(!Number.isFinite(height)||height<0||height>.32)throw new Error('Obstacle height must be 0..0.32 m');
    this.obstacleHeight=height;this.buildObstacles();
  }
  setObstaclesEnabled(on){this.obstaclesEnabled=!!on;}
  setCourse(course){if(!['course','stairs','sequence'].includes(course))throw new Error('Unknown obstacle course');this.course=course;this.length=course==='sequence'?24:8;this.buildObstacles();}
  setStepHeight(height){
    if(!Number.isFinite(height)||height<0||height>.18)throw new Error('Step height must be 0..0.18 m');
    this.stepHeight=height;this.buildObstacles();
  }
  get activeHeight(){return this.course==='sequence'?Math.max(.5,this.stepHeight,this.obstacleHeight):this.course==='stairs'?this.stepHeight:this.obstacleHeight;}
  wrap(z){return ((z%this.length)+this.length)%this.length;}
  isActive(o){return this.obstaclesEnabled&&(o.kind==='wall'||(o.kind==='stair'?this.stepHeight:this.obstacleHeight)>0);}
  buildObstacles(){
    this.obstacles=(this.course==='sequence'?SEQUENCE:this.course==='stairs'?STAIRS:COURSE).map(spec=>{
      const {kind,x,z,width,length,scale}=spec,halfX=width/2,halfZ=length/2;
      const base=Math.max(...[-halfX,0,halfX].flatMap(dx=>[-halfZ,0,halfZ].map(dz=>this.baseHeight(x+dx,z+dz))));
      const bottom=kind==='stair'||kind==='wall'?-.14:base-.14,top=kind==='stair'?this.baseHeight(0,z)+this.stepHeight*spec.level:kind==='wall'?this.baseHeight(0,z)+spec.height:base+this.obstacleHeight*scale,triangles=[],tops=[];
      const add=(a,b,c,walkable=false)=>{const t=[a,b,c];triangles.push(t);if(walkable)tops.push(t);};
      if(kind==='block'||kind==='stair'||kind==='wall'){
        const corners=[[-halfX,-halfZ],[halfX,-halfZ],[halfX,halfZ],[-halfX,halfZ]];
        const lo=corners.map(([dx,dz])=>[x+dx,bottom,z+dz]),hi=corners.map(([dx,dz])=>[x+dx,top,z+dz]);
        add(hi[0],hi[2],hi[1],true);add(hi[0],hi[3],hi[2],true);
        for(let i=0;i<4;i++){const j=(i+1)%4;add(lo[i],hi[i],lo[j]);add(lo[j],hi[i],hi[j]);}
      }else{
        const lo=[],hi=[];
        for(let i=0;i<8;i++){const a=i*Math.PI/4,r=1+.08*Math.sin(i*2.7);lo.push([x+halfX*Math.cos(a)*r,bottom,z+halfZ*Math.sin(a)*r]);hi.push([x+halfX*Math.cos(a)*.72*r,base+this.obstacleHeight*scale*(.66+.1*Math.sin(i*1.8)),z+halfZ*Math.sin(a)*.72*r]);}
        const peak=[x-.06,top,z+.04];
        for(let i=0;i<8;i++){const j=(i+1)%8;add(lo[i],hi[i],lo[j],true);add(lo[j],hi[i],hi[j],true);add(hi[i],peak,hi[j],true);}
      }
      const vertices=triangles.flat();return {...spec,base,top,triangles,tops,minX:Math.min(...vertices.map(p=>p[0])),maxX:Math.max(...vertices.map(p=>p[0])),minZ:Math.min(...vertices.map(p=>p[2])),maxZ:Math.max(...vertices.map(p=>p[2]))};
    });
  }
  height(x,z){
    let h=this.baseHeight(x,z);if(!this.obstaclesEnabled||this.activeHeight===0)return h;
    const zz=this.wrap(z);
    for(const obstacle of this.obstacles){if(!this.isActive(obstacle)||x<obstacle.minX-1e-9||x>obstacle.maxX+1e-9||zz<obstacle.minZ-1e-9||zz>obstacle.maxZ+1e-9)continue;for(const t of obstacle.tops)h=Math.max(h,surfaceHeight(t,x,zz));}
    return h;
  }
  rootHeight(x,z){
    if(!this.obstaclesEnabled||!['stairs','sequence'].includes(this.course))return this.height(x,z);
    // A continuous body path; contact and clearance queries still use the discrete rendered treads.
    const zz=this.wrap(z);let h=this.baseHeight(x,z);
    for(const kind of ['stair','wall']){
      const obstacles=this.obstacles.filter(o=>o.kind===kind&&this.isActive(o));if(!obstacles.length)continue;
      const first=obstacles[0],last=obstacles.at(-1),margin=kind==='wall'?.9:.275;
      const start=first.minZ-margin,end=last.maxZ+margin;
      if(Math.abs(x)>first.width/2||zz<=start||zz>=end)continue;
      const tops=kind==='wall'?[[first.minZ,first.top],[first.maxZ,first.top]]:obstacles.map(o=>[o.z,o.top]);
      const points=[[start,this.baseHeight(x,start)],...tops,[end,this.baseHeight(x,end)]];
      const i=points.findIndex(p=>p[0]>=zz),[a,b]=[points[i-1],points[i]],u=(zz-a[0])/(b[0]-a[0]);
      h=Math.max(h,a[1]*(1-u)+b[1]*u);
    }
    return h;
  }
  stage(z){if(this.course!=='sequence')return this.course;const distance=this.wrap(-z);return distance<7.5?'Stairs & platform':distance<16?'Uneven rocks':distance<21?'50 cm wall':'Return';}
  obstacleTriangles(){return this.obstacles.filter(o=>this.isActive(o)).flatMap(o=>o.triangles.flat(2));}
  obstacleColors(){
    if(!this.obstaclesEnabled||this.activeHeight===0)return [];
    return this.obstacles.filter(o=>this.isActive(o)).flatMap(o=>o.triangles.flatMap(t=>{
      const top=t.every(p=>p[1]>o.base),color=o.kind==='wall'?(top?[.63,.45,.29]:[.43,.28,.18]):o.kind==='rock'?[.24,.29,.33]:o.kind==='stair'?(top?(o.level%2?[.48,.53,.55]:[.55,.59,.60]):[.29,.34,.37]):top?[.49,.40,.29]:[.32,.28,.23];
      return [color,color,color].flat();
    }));
  }
  triangles() {
    const out=[],p=(i,j)=>[-2+i*4/this.nx,this.rows[j][i],j*this.length/this.nz];
    for(let j=0;j<this.nz;j++)for(let i=0;i<this.nx;i++)out.push(...p(i,j),...p(i+1,j+1),...p(i+1,j),...p(i,j),...p(i,j+1),...p(i+1,j+1));
    return out;
  }
}
