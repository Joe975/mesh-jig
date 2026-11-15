import {SphereGeometry,ShaderMaterial,Mesh,BackSide,Color,Vector3} from './vendor/three.module.js';

// One small inward-facing sky dome, no image download or environment-map generation.
export function createSky(){
  const material=new ShaderMaterial({side:BackSide,depthWrite:false,depthTest:false,toneMapped:false,
    uniforms:{zenith:{value:new Color('#598bc4')},horizon:{value:new Color('#ead8b8')},earth:{value:new Color('#798b78')},sunDirection:{value:new Vector3(-.54,.10,.84).normalize()}},
    vertexShader:'varying vec3 ray; void main(){ray=position;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}',
    fragmentShader:`
      varying vec3 ray;
      uniform vec3 zenith,horizon,earth,sunDirection;
      void main(){
        vec3 d=normalize(ray);
        vec3 color=mix(horizon,zenith,smoothstep(0.0,0.22,d.y));
        float azimuth=atan(d.z,d.x);
        float ridge=0.012+0.014*sin(azimuth*7.0)+0.008*sin(azimuth*13.0);
        float land=1.0-smoothstep(ridge-0.01,ridge+0.015,d.y);
        color=mix(color,earth,land*0.83);
        float sun=dot(d,sunDirection);
        color+=vec3(0.06,0.045,0.02)*pow(max(sun,0.0),32.0);
        color=mix(color,vec3(1.0,0.83,0.53),smoothstep(0.9988,0.9994,sun));
        float cloud=sin(azimuth*5.0+d.y*34.0)+0.5*sin(azimuth*13.0-d.y*19.0);
        float band=smoothstep(0.06,0.15,d.y)*(1.0-smoothstep(0.28,0.47,d.y));
        color=mix(color,vec3(0.9,0.91,0.9),smoothstep(0.75,1.2,cloud)*band*0.35);
        gl_FragColor=vec4(color,1.0);
        #include <colorspace_fragment>
      }`});
  const sky=new Mesh(new SphereGeometry(40,24,12),material);sky.renderOrder=-1000;sky.frustumCulled=false;return sky;
}
